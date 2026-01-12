import numpy as np


from .types import UpdateLog
from .components import Node
from .aptree import APTree
from concurrent.futures import ThreadPoolExecutor


class APForest:
    """
    An ensemble of APTrees for clustering data points.
    """

    def __init__(
        self,
        *,
        n_trees: int = 32,
        leaf_max_size: int = 256,
        b_strategy: str = "default",
        seed: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.leaf_max_size = leaf_max_size
        self.b_strategy = b_strategy
        self.seed = seed

        self.trees: list[APTree] = [
            APTree(
                leaf_max_size=self.leaf_max_size,
                b_strategy=self.b_strategy,
                seed=self.seed + i,
            )
            for i in range(self.n_trees)
        ]
        self.cluster_update_initialized = False

        self._last_update_logs: list[UpdateLog] = []

    def insert(self, batch: np.ndarray) -> None:
        self._last_update_logs = []

        def insert_tree(tree):
            tree.insert(batch)
            return tree.get_update_log()

        with ThreadPoolExecutor(max_workers=16) as executor:
            self._last_update_logs = list(executor.map(insert_tree, self.trees))

    def get_leaf_nodes(self) -> list[list[Node]]:
        return [tree.get_leaf_nodes() for tree in self.trees]

    def get_last_update_logs(self) -> list[UpdateLog]:
        return self._last_update_logs
