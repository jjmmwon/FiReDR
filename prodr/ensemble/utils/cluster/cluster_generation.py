import numpy as np
import scipy.sparse as sp

from prodr.ensemble.components import MicroCluster


def generate_micro_clusters(
    n_components: int,
    labels: np.ndarray,
    cooccurr_mtx: sp.csr_array,
    global_ids: list[int] | np.ndarray,
    head_global_idx: int | None = None,
) -> list[MicroCluster]:
    """
    Generate micro-clusters based on the result of connected_components.

    Args:
        n_components (int): Number of micro-clusters to generate.
        labels (np.ndarray): Array of labels for data points.
    """

    global_ids_by_label: dict[int, list[int]] = {
        label: [] for label in range(n_components)
    }

    local_ids_by_label: dict[int, list[int]] = {
        label: [] for label in range(n_components)
    }

    head_label: int | None = None
    for idx, label in enumerate(labels):
        label = int(label)
        if global_ids[idx] == head_global_idx:
            head_label = label
        global_ids_by_label[label].append(int(global_ids[idx]))
        local_ids_by_label[label].append(idx)

    return [
        MicroCluster(
            indices=global_ids_by_label[i],
            cooccurrence_count=cooccurr_mtx[local_ids_by_label[i]][
                :, local_ids_by_label[i]
            ],
            head=(
                head_global_idx
                if head_label == i and head_global_idx is not None
                else global_ids_by_label[i][0]
            ),
        )
        for i in range(n_components)
    ]
