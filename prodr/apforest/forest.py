import numpy as np

from prodr.apforest import APTree
from prodr.components import Node


class APCForest:
    """
    Adaptive Progressive Clustering Forest for clustering high-dimensional streaming data
    """

    def __init__(
        self,
        n_trees: int = 32,
        *,
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

        self._X: np.ndarray | None = None
        self._n_features: int = -1
        self._idx_counter: int = 0

    def _ensure_initialized(self, batch: np.ndarray) -> None:
        if self._n_features != -1:
            return
        self._n_features = batch.shape[1]

    def insert(self, batch: np.ndarray) -> None:
        """
        Insert new data points into the forest.
        Args:
            X (np.ndarray): New data points to be inserted.
        """
        self._ensure_initialized(batch)

        if self._X is None:
            self._X = batch
        else:
            self._X = np.vstack([self._X, batch])

        for tree in self.trees:
            tree.insert(batch)

        self._idx_counter += batch.shape[0]

    def get_leaf_nodes(self) -> list[list[Node]]:
        """
        Get the current leaf nodes of all trees in the forest.
        Returns:
            list[list[Node]]: A list containing lists of current leaf nodes for each tree.
        """
        return [tree.get_leaf_nodes() for tree in self.trees]
