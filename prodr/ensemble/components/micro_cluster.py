from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp


@dataclass
class MicroCluster:
    """
    A micro-cluster representing a small cluster of data points.

    Attributes:
        indices (list[int]): List of data point indices in the micro-cluster.
        cooccurrence_count (sp.csr_array): Cooccurrence count 2D matrix for the data points in the micro-cluster.
        head (np.ndarray | None): Representative data point (head) of the micro-cluster.
        gidx_to_lidx (dict[int, int]): Mapping from global indices to local indices within the micro-cluster.
    """

    indices: list[int]
    cooccurrence_count: sp.csr_array
    head: int
    gidx_to_lidx: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.indices) != len(set(self.indices)):  # type: ignore
            raise ValueError("Indices contain duplicate entries.")
        self.generate_gidx_to_lidx_mapping()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    @property
    def size(self) -> int:
        return len(self.indices)

    def generate_gidx_to_lidx_mapping(self) -> None:
        self.gidx_to_lidx = {
            idx: local_idx for local_idx, idx in enumerate(self.indices)
        }

    def get_local_idx(self, global_idx: int) -> int:
        if global_idx in self.gidx_to_lidx:
            return self.gidx_to_lidx[global_idx]
        else:
            raise KeyError(f"Global index {global_idx} not found in micro-cluster.")

    def get_local_indices(self, global_indices: list[int]) -> list[int]:
        return [self.get_local_idx(gidx) for gidx in global_indices]

    def update_cooccurrence_count(
        self,
        gid_rows: list[int] | int,
        gid_cols: list[int] | int,
        counts: list[int] | int,
    ):
        # type check whether inputs are same types and they are lists and have same lengths

        if (
            isinstance(gid_rows, int)
            and isinstance(gid_cols, int)
            and isinstance(counts, int)
        ):
            gid_rows = [gid_rows]
            gid_cols = [gid_cols]
            counts = [counts]

        elif (
            isinstance(gid_rows, list)
            and isinstance(gid_cols, list)
            and isinstance(counts, list)
        ):
            if not len(gid_rows) == len(gid_cols) == len(counts):
                raise ValueError("Input lists must have the same length.")
        else:
            raise TypeError("Input types must be all int or all list of int.")

        lidx_rows = self.get_local_indices(gid_rows)
        lidx_cols = self.get_local_indices(gid_cols)

        # Convert to COO format for efficient bulk assignment
        coo = self.cooccurrence_count.tocoo()

        # Add new entries
        coo.row = np.concatenate([coo.row, lidx_rows, lidx_cols])
        coo.col = np.concatenate([coo.col, lidx_cols, lidx_rows])
        coo.data = np.concatenate([coo.data, counts, counts])

        # Convert back to CSR format
        self.cooccurrence_count = coo.tocsr()
        self.cooccurrence_count.sum_duplicates()
        self.cooccurrence_count[self.cooccurrence_count < 0] = 0
        self.cooccurrence_count.eliminate_zeros()

    def is_dirty(self, threshold: int) -> bool:
        return bool((self.cooccurrence_count.data < threshold).any())

    def split_micro_cluster(self, threshold: int) -> list["MicroCluster"]:
        adjacency_matrix = self.cooccurrence_count >= threshold

        filtered_inner_structure = self.cooccurrence_count.multiply(adjacency_matrix)
        filtered_inner_structure.eliminate_zeros()

        n_components, labels = sp.csgraph.connected_components(
            adjacency_matrix, directed=False, return_labels=True
        )

        gids_by_label: dict[int, list[int]] = {
            label: [] for label in range(n_components)
        }
        lids_by_label: dict[int, list[int]] = {
            label: [] for label in range(n_components)
        }

        for idx, label in enumerate(labels):
            gids_by_label[label].append(self.indices[idx])
            lids_by_label[label].append(idx)

        return [
            MicroCluster(
                indices=gids_by_label[i],
                cooccurrence_count=filtered_inner_structure[lids_by_label[i]][
                    :, lids_by_label[i]
                ],
                head=(
                    self.head
                    if labels[self.get_local_idx(self.head)] == i
                    else gids_by_label[i][0]
                ),
            )
            for i in range(n_components)
        ]

    def merge_micro_clusters(
        self, micro_clusters: list["MicroCluster"]
    ) -> "MicroCluster":
        for mc in micro_clusters:
            self.indices += mc.indices

        matrices = [self.cooccurrence_count] + [
            mc.cooccurrence_count for mc in micro_clusters
        ]

        self.cooccurrence_count = sp.block_diag(matrices, format="csr")  # type: ignore

        self.generate_gidx_to_lidx_mapping()

        return self
