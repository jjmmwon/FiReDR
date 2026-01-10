import numpy as np

from .types import UpdateLog
from .aptree import APTree
from .cluster_tracker import ClusterTracker


class Ensemble:
    """
    Ensemble of Adaptive Progressive Trees for clustering high-dimensional streaming data
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
        self.cluster_update_initialized = False

    def _ensure_initialized(self, X: np.ndarray) -> None:
        if self._n_features != -1:
            return
        self._n_features = X.shape[1]

    def _ensure_cluster_initialized(self) -> None:
        if self.cluster_update_initialized:
            return
        assert self._X is not None

        if len(self.trees[0].get_leaf_nodes()) > 8:
            self.cluster_tracker.initialization(
                E=[tree.get_leaf_nodes() for tree in self.trees],
                n_samples=self._X.shape[0],
            )
            self.cluster_update_initialized = True

    def insert(self, batch: np.ndarray) -> None:
        self._ensure_initialized(batch)
        self._X = batch if self._X is None else np.vstack([self._X, batch])

        update_logs: list[UpdateLog] = []
        for tree in self.trees:
            tree.insert(batch)
            update_logs.append(tree.get_update_log())

        self.update_clusters(update_logs)

    def update_clusters(self, update_logs: list[UpdateLog]) -> None:
        self._ensure_cluster_initialized()

        if not self.cluster_update_initialized:
            return

        # self.cluster_tracker.update(update_logs)
