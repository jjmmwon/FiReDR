import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from prodr.ensemble.components import MicroCluster


def extract_micro_clusters(
    collision_matrix: sp.csr_matrix, threshold: int
) -> tuple[list[np.ndarray], list[sp.csr_matrix]]:
    """
    Split micro-clusters based on the collision matrix and a given threshold.

    Args:
        collision_matrix (sp.csr_matrix): The collision matrix where C[i, j] indicates
                                          the number of trees in which data points i and j
                                            share the same leaf node.
        threshold (int): The minimum number of collisions required to consider two data points

                            as belonging to the same micro-cluster.
    Returns:
        tuple[list[np.ndarray], list[sp.csr_matrix]]: A tuple containing:
            - A list of numpy arrays, each representing the indices of data points in a micro-cluster.
            - A list of sparse matrices, each representing the collision matrix of a micro-cluster.
    """

    # Create a binary adjacency matrix based on the threshold
    adjacency_matrix = collision_matrix >= threshold

    # Find connected components in the graph represented by the adjacency matrix
    n_components, labels = connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )
    label_to_indices: dict[int, list[int]] = {
        label: [] for label in range(n_components)
    }
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    members = [np.array(indices) for indices in label_to_indices.values()]
    micro_clusters = [collision_matrix[indices][:, indices] for indices in members]

    return members, micro_clusters


def merge_micro_clusters(
    mirco_cluster1: MicroCluster, micro_cluster2: MicroCluster
) -> MicroCluster:
    members1, MC1 = (mirco_cluster1.data_indices, mirco_cluster1.inner_structure)
    members2, MC2 = (micro_cluster2.data_indices, micro_cluster2.inner_structure)

    merged_members = merge_members(members1, members2)

    id2pos = {idx: pos for pos, idx in enumerate(merged_members)}
    K = merged_members.size

    MC1_global = lift_to_global(MC1, members1, id2pos, K)
    MC2_global = lift_to_global(MC2, members2, id2pos, K)

    merged_inner_structure = MC1_global + MC2_global

    return MicroCluster(
        data_indices=merged_members,
        inner_structure=merged_inner_structure,
    )


def merge_members(member1: np.ndarray, member2: np.ndarray) -> np.ndarray:
    seen = set(member1)
    merged = list(member1)
    for idx in member2:
        if idx not in seen:
            merged.append(idx)
            seen.add(idx)
    return np.array(merged)


def lift_to_global(
    micro_cluster: sp.csr_matrix,
    members: np.ndarray,
    id2pos: dict[int, int],
    K: int,
) -> sp.csr_matrix:
    ids_local = np.asarray(members)
    idx = np.fromiter(
        (id2pos[id_] for id_ in ids_local), count=ids_local.size, dtype=np.int64
    )

    MC_coo = micro_cluster.tocoo()
    rows_g = idx[MC_coo.row]
    cols_g = idx[MC_coo.col]
    data = MC_coo.data

    return sp.csr_matrix((data, (rows_g, cols_g)), shape=(K, K))
