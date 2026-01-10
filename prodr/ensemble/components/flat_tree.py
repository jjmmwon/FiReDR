from dataclasses import dataclass, field

import numpy as np

from .node import Node


@dataclass
class FlatTree:
    """
    A flattened representation of an adaptive partitioning tree.

    Attributes:
        root (Node): The root node of the tree.
        root_id (int): The identifier for the root node.
        left (np.ndarray): Array of left child node indices.
        right (np.ndarray): Array of right child node indices.
        thresholds (np.ndarray): Array of hyperplane thresholds.
        depth (np.ndarray): Array of node depths.
        id_to_node (list[Node]): Mapping from node IDs to Node objects.
    """

    root: Node
    root_id: int = 0

    left: list[int] = field(default_factory=lambda: [-1])
    right: list[int] = field(default_factory=lambda: [-1])
    thresholds: list[float | np.float64] = field(default_factory=lambda: [np.nan])
    depth: list[int] = field(default_factory=lambda: [0])

    id_to_node: list[Node] = field(default_factory=lambda: [])
    node_to_id: dict[int, int] = field(default_factory=lambda: {})

    def __post_init__(self):
        self.id_to_node.append(self.root)
        self.node_to_id[id(self.root)] = self.root_id

    def insert_node(self, node: Node) -> int:
        """
        Insert a node into the flattened tree representation.

        Args:
            node (Node): The node to be inserted.

        Returns:
            np.int64: The ID assigned to the inserted node.
        """
        node_id = len(self.id_to_node)
        self.id_to_node.append(node)
        self.node_to_id[id(node)] = node_id

        self.left.append(-1)
        self.right.append(-1)
        self.thresholds.append(np.nan)
        self.depth.append(node.depth)

        return node_id

    def split_node(
        self, node: Node, left_node: Node, right_node: Node, threshold: float
    ) -> None:
        """
        Split a node into left and right children in the flattened tree representation.

        Args:
            node_id (np.int64): The ID of the node to be split.
            left_id (np.int64): The ID of the left child node.
            right_id (np.int64): The ID of the right child node.
            threshold (float): The hyperplane threshold for the split.
        """
        node_id = self.node_to_id[id(node)]
        left_id = self.insert_node(left_node)
        right_id = self.insert_node(right_node)

        self.left[node_id] = left_id
        self.right[node_id] = right_id
        self.thresholds[node_id] = threshold
