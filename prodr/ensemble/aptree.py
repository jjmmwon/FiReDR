from collections import deque

import numpy as np

from .components import Node, ProgressiveDataStorage
from .types import InsertionEvent, NodeSplitEvent
from .utils import (
    generate_hyperplane,
    generate_normal,
    split_node,
    # traverse_to_leaf,
)


class APTree:
    """
    Adaptive Progressive Tree for clustering high-dimensional streaming data
    """

    def __init__(
        self,
        *,
        data: ProgressiveDataStorage,
        leaf_max_size: int = 256,
        b_strategy: str = "default",
        seed: int = 42,
    ) -> None:
        self.data = data
        self.leaf_max_size = leaf_max_size
        self.b_strategy = b_strategy
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self._root: Node = Node(
            indices=[],
            depth=0,
        )
        self._leaf_nodes: deque[Node] = deque([self._root])
        self._id_to_node: list[Node] = []

        self.normals: np.ndarray = np.array([])

    def _init_normal(self, n_features: int) -> None:
        self.normals = generate_normal(n_features, self._rng).reshape(1, -1)

    def insert(self, start_idx: int) -> list[InsertionEvent]:
        """
        Insert new data points into the tree.
        Args:
            data (ProgressiveDataStorage): New data points to be inserted.
        """
        batch = self.data[start_idx:]
        insertion_events = self._insert_batch(batch, start_idx)

        return insertion_events

    def split(self) -> list[NodeSplitEvent]:
        split_events = self._split_nodes()

        return split_events

    def _insert_batch(self, batch: np.ndarray, start_idx: int) -> list[InsertionEvent]:
        """"""
        insertion_events: list[InsertionEvent] = []

        if self.normals.size == 0:
            self._init_normal(self.data.n_features)  # type: ignore

        projections = batch @ self.normals.T
        for i in range(batch.shape[0]):
            leaf_node = self._traverse_to_leaf(projections[i])
            leaf_node.indices.append(start_idx + i)
            insertion_events.append(
                InsertionEvent(data_index=start_idx + i, node=leaf_node)
            )
            self._id_to_node.append(leaf_node)
        return insertion_events

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
            node = node.left if projection[node.depth] >= offset else node.right

        return node

    def _split_nodes(self) -> list[NodeSplitEvent]:
        """
        Split a leaf node into two child nodes based on a hyperplane.
        Args:
            node (Node): The leaf node to be split.
        """

        leaf_nodes: list[Node] = []
        split_events: list[NodeSplitEvent] = []

        while self._leaf_nodes:
            node = self._leaf_nodes.popleft()
            if len(node.indices) <= self.leaf_max_size:
                leaf_nodes.append(node)
                continue

            data = self.data[node.indices]
            normal_vector = (
                self.normals[node.depth] if node.depth < self.normals.shape[0] else None
            )
            hyperplane = generate_hyperplane(
                data=data, normal_vector=normal_vector, rng=self._rng
            )
            if normal_vector is None:
                self.normals = np.vstack([self.normals, hyperplane.normal])

            idx_arr = np.array(node.indices)
            mask = np.dot(data, hyperplane.normal) >= hyperplane.offset

            left_idx = idx_arr[mask]
            right_idx = idx_arr[np.logical_not(mask)]

            left_node, right_node = split_node(
                node=node,
                left=left_idx.tolist(),
                right=right_idx.tolist(),
                hyperplane=hyperplane,
            )

            self._leaf_nodes.append(left_node)
            self._leaf_nodes.append(right_node)

            for idx in left_idx:
                self._id_to_node[idx] = left_node

            for idx in right_idx:
                self._id_to_node[idx] = right_node

            split_events.append(
                NodeSplitEvent(
                    parent_node=node, left_child=left_node, right_child=right_node
                )
            )

        self._leaf_nodes = deque(leaf_nodes)

        return split_events

    def get_id_to_node_mapping(self) -> list[Node]:
        return self._id_to_node

    def get_node_by_id(self, idx: int) -> Node:
        return self._id_to_node[idx]

    def get_leaf_nodes(self) -> list[Node]:
        return list(self._leaf_nodes)
