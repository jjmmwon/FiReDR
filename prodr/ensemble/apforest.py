from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .types import InsertionEvent, NodeSplitEvent
from .components import Node, ProgressiveDataStorage
from .aptree import APTree


class APForest:
    """
    An ensemble of APTrees for clustering data points.
    """

    def __init__(
        self,
        *,
        data: ProgressiveDataStorage,
        n_trees: int = 32,
        leaf_max_size: int = 256,
        b_strategy: str = "default",
        seed: int = 42,
    ) -> None:
        self.data = data
        self.n_trees = n_trees
        self.leaf_max_size = leaf_max_size
        self.b_strategy = b_strategy
        self.seed = seed

        self.trees: list[APTree] = [
            APTree(
                data=self.data,
                leaf_max_size=self.leaf_max_size,
                b_strategy=self.b_strategy,
                seed=self.seed + i,
            )
            for i in range(self.n_trees)
        ]

    def insert(self, start_idx: int) -> list[list[InsertionEvent]]:
        insertion_events = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            insertion_events = list(
                executor.map(lambda tree: tree.insert(start_idx), self.trees)
            )

        return insertion_events

    def split(self) -> list[list[NodeSplitEvent]]:
        split_events = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            split_events = list(executor.map(lambda tree: tree.split(), self.trees))

        return split_events

    def get_id_to_node_mappings(self) -> list[list[Node]]:
        id_to_node_mappings = [tree.get_id_to_node_mapping() for tree in self.trees]

        return id_to_node_mappings

    def get_all_leaf_nodes(self) -> list[list[Node]]:
        leaf_nodes = [tree.get_leaf_nodes() for tree in self.trees]

        return leaf_nodes
