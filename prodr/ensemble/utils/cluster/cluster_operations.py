import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


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
