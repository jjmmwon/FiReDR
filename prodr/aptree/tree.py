from typing import Optional, Tuple
import numpy as np

from prodr.aptree.structures import Node
from prodr.aptree.types import Hyperplane


class APTree:
    """
    Adaptive Progressive Tree for clustering high-dimensional streaming data
    """

    def __init__(
        self, *, leaf_max_size: int = 256, b_strategy: str = "default", seed: int = 42
    ) -> None:
        self.leaf_max_size = leaf_max_size
        self.b_strategy = b_strategy
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)
        self._depth_to_a: dict[int, np.ndarray] = {}

        self._root: Node = Node(
            data_indices=[],
            depth=0,
        )

        self._X: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def insert(self, X: np.ndarray) -> None:
        """
        Insert new data points into the tree.
        Args:
            X (np.ndarray): New data points to be inserted.
        """
        idx_start = 0 if self._X is None else self._X.shape[0]

        self._X = X if self._X is None else np.vstack((self._X, X))

        if self._n_features is None:
            self._n_features = X.shape[1]
        elif self._n_features != X.shape[1]:
            raise ValueError("All data points must have the same number of features.")

        for x_index in range(X.shape[0]):
            self._insert_point(self._root, idx_start + x_index)

    def _insert_point(self, node: Node, x_index: int) -> None:
        """
        Insert a single data point into the tree starting from the given node.
        Args:
            node (Node): The current node in the tree.
            x (np.ndarray): The data point to be inserted.
        """
        assert self._X is not None
        x = self._X[x_index]

        if node.is_leaf:
            node.data_indices.append(x_index)
            if len(node.data_indices) >= self.leaf_max_size:
                self._split_node(node)
        else:
            if node.hyperplane is None:
                raise ValueError("Non-leaf nodes must have a hyperplane.")

            assert node.left is not None and node.right is not None

            if node.hyperplane.evaluate(x) <= 0:
                self._insert_point(node.left, x_index)
            else:
                self._insert_point(node.right, x_index)

    def _split_node(self, node: Node) -> None:
        """
        Split a leaf node into two child nodes based on a hyperplane.
        Args:
            node (Node): The leaf node to be split.
        """
        hyperplane, left, right = self._generate_hyperplane(
            node.data_indices, node.depth
        )

        left_node = Node(
            parent=node,
            data_indices=left.tolist(),
            depth=node.depth + 1,
        )

        right_node = Node(
            parent=node,
            data_indices=right.tolist(),
            depth=node.depth + 1,
        )

        node.is_leaf = False
        node.hyperplane = hyperplane
        node.left = left_node
        node.right = right_node

    def _generate_hyperplane(
        self, data_indices: list[int], depth: int
    ) -> Tuple[Hyperplane, np.ndarray, np.ndarray]:
        """
        Generate a hyperplane to split the data points in the node.
        Args:
            data_indices (list[int]): Indices of data points in the node.
            depth (int): Depth of the node in the tree.
        Returns:
            Hyperplane: The generated hyperplane.
        """
        assert self._X is not None

        if depth not in self._depth_to_a:
            a = np.random.normal(size=self._n_features)
            a /= np.linalg.norm(a)
            self._depth_to_a[depth] = a
        else:
            a = self._depth_to_a[depth]

        projections = np.dot(self._X[data_indices], a)
        offset = np.median(projections)

        left = np.array(data_indices)[projections <= offset]
        right = np.array(data_indices)[projections > offset]

        return Hyperplane(normal=a, offset=offset), left, right

    def get_leaf_nodes(self) -> list[Node]:
        """
        Retrieve all leaf nodes in the tree.
        Returns:
            list[Node]: List of leaf nodes.
        """
        leaves: list[Node] = []

        def _traverse(node: Node) -> None:
            if node.is_leaf:
                leaves.append(node)
            else:
                assert node.left is not None and node.right is not None
                _traverse(node.left)
                _traverse(node.right)

        _traverse(self._root)
        return leaves
