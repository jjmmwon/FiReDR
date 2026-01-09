import numpy as np
from prodr.apforest import APTree
from prodr.components import Node

from .cluster_tracker import ClusterTracker


class Ensemble:
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

        self._X: np.ndarray | None = None
        self._n_features: int = -1

        self.trees: list[APTree] = [
            APTree(
                leaf_max_size=self.leaf_max_size,
                b_strategy=self.b_strategy,
                seed=self.seed + i,
            )
            for i in range(self.n_trees)
        ]
        self.cluster_tracker = ClusterTracker()

    def _ensure_initialized(self, X: np.ndarray) -> None:
        if self._n_features != -1:
            return
        self._n_features = X.shape[1]

    def insert(self, batch: np.ndarray) -> None:
        self._ensure_initialized(batch)
        self._X = batch if self._X is None else np.vstack([self._X, batch])

        for tree in self.trees:
            tree.insert(batch)

    def update_clusters(self) -> None:
        pass

    def get_leaf_nodes(self) -> list[list[Node]]:
        return [tree.get_leaf_nodes() for tree in self.trees]
