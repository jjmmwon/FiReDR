import numpy as np

from numba import njit, prange

from prodr.components import FlatTree, Node


def traverse_to_leaf(flat_tree: FlatTree, projections: np.ndarray) -> list[Node]:
    """
    Traverse the flattened tree to find the appropriate leaf nodes for given data points.

    Args:
        flat_tree (FlatTree): The flattened tree representation.
        projections (np.ndarray): The projections of the data points onto the normal vectors.

    Returns:
        np.ndarray: An array of leaf node IDs corresponding to each data point.
    """
    leaf_ids = _traverse_to_leaf(
        projections,
        np.array(flat_tree.left, dtype=np.int64),
        np.array(flat_tree.right, dtype=np.int64),
        np.array(flat_tree.thresholds, dtype=np.float64),
        np.array(flat_tree.depth, dtype=np.int64),
        flat_tree.root_id,
    )
    return _leaf_ids_to_Node(flat_tree, leaf_ids)


@njit(parallel=True)
def _traverse_to_leaf(
    projections: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
    depths: np.ndarray,
    root_id: int,
) -> np.ndarray:

    n_samples: int = projections.shape[0]
    leaf_ids = np.empty(n_samples, dtype=np.int64)

    for i in prange(n_samples):  # pylint: disable=not-an-iterable
        node_id = root_id

        while left[node_id] != -1 and right[node_id] != -1:
            depth = depths[node_id]
            offset = offsets[node_id]
            projection = projections[i, depth]

            if projection >= offset:
                node_id = left[node_id]
            else:
                node_id = right[node_id]

        leaf_ids[i] = node_id

    return leaf_ids


def _leaf_ids_to_Node(flat_tree: FlatTree, leaf_ids: np.ndarray) -> list[Node]:
    """
    Convert an array of leaf node IDs to their corresponding Node objects.

    Args:
        flat_tree (FlatTree): The flattened tree representation.
        leaf_ids (np.ndarray): An array of leaf node IDs.

    Returns:
        list[Node]: A list of Node objects corresponding to the leaf IDs.
    """
    return [flat_tree.id_to_node[leaf_id] for leaf_id in leaf_ids]
