import numpy as np
import scipy.sparse as sp

from prodr.ensemble.components import Node
from prodr.ensemble.types import InsertionEvent


def generate_collision_matrix_single(
    leaf_node_list: list[Node], n_samples: int
) -> sp.csr_matrix:
    """
    Generate a collision matrix C where C[i, j] indicates the number of trees
    in which data points i and j share the same leaf node.
    Args:
        leaf_node_list (list[Node]): List of leaf nodes from a tree.
        n_samples (int): Total number of data points.
    Returns:
        sp.csr_matrix: The collision matrix of shape (n_samples, n_samples).
    """
    set_index = np.full(n_samples, -1, dtype=np.int64)
    for j, leaf_node in enumerate(leaf_node_list):
        set_index[leaf_node.data_indices] = j

    rows = np.arange(n_samples, dtype=np.int64)
    cols = set_index
    data = np.ones(n_samples, dtype=np.int8)
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, len(leaf_node_list)))

    C = (M @ M.T).tocsr()

    C.setdiag(0)
    C.eliminate_zeros()

    return C


def generate_collision_matrix(
    ensemble_leaf_nodes: list[list[Node]], n_samples: int
) -> sp.csr_matrix:
    """
    Generate a collision matrix C where C[i, j] indicates the number of trees
    in which data points i and j share the same leaf node across an ensemble of trees.
    Args:
        ensemble_leaf_nodes (list[list[Node]]): List of leaf nodes from each tree in the ensemble.
        n_samples (int): Total number of data points.
    Returns:
        sp.csr_matrix: The collision matrix of shape (n_samples, n_samples).
    """
    C_total = sp.csr_matrix((n_samples, n_samples), dtype=np.int16)

    for leaf_node_list in ensemble_leaf_nodes:
        C_tree = generate_collision_matrix_single(leaf_node_list, n_samples)
        C_total += C_tree

    return C_total


def update_newly_inserted_points(
    insertion_logs: list[list[InsertionEvent]],
    idx_range: tuple[int, int],
):
    if len(insertion_logs[0]) != (idx_range[1] - idx_range[0]):
        raise ValueError(
            "Length of insertion_logs must match the number of newly inserted points."
        )
