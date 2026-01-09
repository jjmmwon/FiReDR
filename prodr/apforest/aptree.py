from collections import deque

import numpy as np

from prodr.components import Node, FlatTree
from prodr.apforest.utils import (
    generate_hyperplane,
    generate_normal,
    split_node,
    traverse_to_leaf,
)


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

        self._root: Node = Node(
            data_indices=[],
            depth=0,
        )
        self._flat_tree: FlatTree = FlatTree(root=self._root)
        self._leaf_nodes: deque[Node] = deque([self._root])

        self._X: np.ndarray | None = None
        self._n_features: int = -1
        self._idx_counter: int = 0
        self.normals: np.ndarray = np.array([])

    def _ensure_initialized(self, X: np.ndarray) -> None:
        if self._n_features != -1:
            return
        self._n_features = X.shape[1]
        self.normals = generate_normal(self._n_features, self._rng).reshape(1, -1)

    def insert(self, batch: np.ndarray) -> None:
        """
        Insert new data points into the tree.
        Args:
            X (np.ndarray): New data points to be inserted.
        """
        self._ensure_initialized(batch)
        self._insert_batch(batch)
        self._split_nodes()

    def _insert_batch(self, batch: np.ndarray) -> None:
        """"""
        self._X = batch if self._X is None else np.vstack([self._X, batch])

        projections = batch @ self.normals.T
        for i in range(batch.shape[0]):
            leaf_node = self._traverse_to_leaf(projections[i])
            leaf_node.data_indices.append(self._idx_counter + i)
        self._idx_counter += batch.shape[0]

    def _traverse_to_leaf(self, projection: np.ndarray) -> Node:
        """
        Traverse the tree to find the appropriate leaf node for a given data point.
        Args:
            projection (np.ndarray): The projection of the data point onto the normal vectors.
        Returns:
            Node: The leaf node where the data point should be inserted.
        """
        node = self._root
        while not node.is_leaf:
            offset = node.hyperplane.offset
            if projection[node.depth] >= offset:
                node = node.left
            else:
                node = node.right

        return node

    def _split_nodes(self) -> None:
        """
        Split a leaf node into two child nodes based on a hyperplane.
        Args:
            node (Node): The leaf node to be split.
        """
        assert self._X is not None

        leaf_nodes: list[Node] = []

        while self._leaf_nodes:
            node = self._leaf_nodes.popleft()
            if len(node.data_indices) <= self.leaf_max_size:
                leaf_nodes.append(node)
                continue

            data = self._X[node.data_indices]
            normal_vector = (
                self.normals[node.depth] if node.depth < self.normals.shape[0] else None
            )
            hyperplane = generate_hyperplane(
                data=data, normal_vector=normal_vector, rng=self._rng
            )
            if normal_vector is None:
                self.normals = np.vstack([self.normals, hyperplane.normal])

            mask = np.dot(data, hyperplane.normal) >= hyperplane.offset
            left = np.array(node.data_indices)[mask]
            right = np.array(node.data_indices)[np.logical_not(mask)]

            left_node, right_node = split_node(
                node=node, left=list(left), right=list(right), hyperplane=hyperplane
            )

            self._leaf_nodes.append(left_node)
            self._leaf_nodes.append(right_node)
            self._flat_tree.split_node(
                node=node,
                left_node=left_node,
                right_node=right_node,
                threshold=hyperplane.offset,
            )

        self._leaf_nodes = deque(leaf_nodes)

    def get_leaf_nodes(self) -> list[Node]:
        """
        Get the current leaf nodes of the tree.
        Returns:
            list[Node]: The list of current leaf nodes.
        """
        return list(self._leaf_nodes)
