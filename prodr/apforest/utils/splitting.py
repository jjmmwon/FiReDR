from typing import Optional
import numpy as np

from prodr.components import Hyperplane, Node


def split_node(
    node: Node, left: list[int], right: list[int], hyperplane: Hyperplane
) -> tuple[Node, Node]:
    """
    Split a node into two child nodes based on the provided hyperplane and data indices.
    Args:
        node (Node): The node to be split.
        left (list[int]): Indices of data points for the left child.
        right (list[int]): Indices of data points for the right child.
        hyperplane (Hyperplane): The hyperplane used for splitting.
    """
    left_child = Node(
        data_indices=left,
        depth=node.depth + 1,
        is_leaf=True,
        parent=node,
    )
    right_child = Node(
        data_indices=right,
        depth=node.depth + 1,
        is_leaf=True,
        parent=node,
    )

    node.left = left_child
    node.right = right_child
    node.is_leaf = False
    node.hyperplane = hyperplane

    return left_child, right_child


def generate_hyperplane(
    data: np.ndarray,
    normal_vector: Optional[np.ndarray] = None,
    rng: np.random.Generator | None = None,
) -> Hyperplane:
    """
    Generate a hyperplane that approximately bisects the given dataset.
    Args:
        data (np.ndarray): The input dataset, shape (n_samples, n_features).
        normal_vector (Optional[np.ndarray]): An optional normal vector for the hyperplane.
        seed (int): Seed for random number generator for reproducibility.
    Returns:
        tuple[np.ndarray, float]: A tuple containing the normal vector and offset of the hyperplane
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_features = data.shape[1]
    normal = (
        normal_vector if normal_vector is not None else generate_normal(n_features, rng)
    )

    projections = np.dot(data, normal)
    offset = np.median(projections)

    return Hyperplane(normal=normal, offset=offset)


def generate_normal(
    n_features: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate a random normal vector for a hyperplane.
    Args:
        n_features (int): The number of features (dimensions).
        seed (int): Seed for random number generator for reproducibility.
    Returns:
        np.ndarray: A random normal vector of shape (n_features,).
    """
    rng = rng if rng is not None else np.random.default_rng()
    normal = rng.normal(size=n_features)
    return normal
