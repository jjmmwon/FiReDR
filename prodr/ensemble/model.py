import numpy as np
from .apforest import APForest
from .cluster_tracker import ClusterTracker


class EnsembleModel:
    def __init__(
        self,
        n_trees: int = 8,
        leaf_max_size: int = 128,
        b_strategy: str = "random",
        seed: int = 42,
    ) -> None:
        self.forest = APForest(
            n_trees=n_trees,
            leaf_max_size=leaf_max_size,
            b_strategy=b_strategy,
            seed=seed,
        )

        self.cluster_tracker = ClusterTracker()
        self._cluster_initialized = False

    def insert(self, batch: np.ndarray) -> None:

        self.forest.insert(batch)
        E = self.forest.get_leaf_nodes()
        last_update_logs = self.forest.get_last_update_logs()

        self.cluster_tracker.insert(E, last_update_logs)
