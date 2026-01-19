from typing import Literal
import numpy as np

from .apforest import APForest
from .cluster_handler import ClusterHandler
from .components import ProgressiveDataStorage, MicroCluster
from .types import ClusterUpdateEvent


class Ensemble:
    """
    An ensemble clustering model using an ensemble of APTrees and micro-cluster management.
    Attributes:
        n_trees (int): Number of trees in the ensemble.
        leaf_max_size (int): Maximum size of leaf nodes in each tree.
        threshold (int): Threshold for micro-cluster operations.
        b_strategy (str): Strategy for generating hyperplanes ("euclidean" or "cosine
        seed (int): Random seed for reproducibility.
        data (ProgressiveDataStorage): Storage for progressive data points.
        forest (APForest): The ensemble of APTrees.
        cluster_handler (EnsembleClusterHandler): Handler for managing micro-clusters.
    """

    def __init__(
        self,
        n_trees: int = 8,
        leaf_max_size: int = 128,
        threshold: int | Literal["default"] = "default",
        b_strategy: Literal["euclidean", "cosine"] = "euclidean",
        seed: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.leaf_max_size = leaf_max_size
        self.threshold: int = threshold if threshold != "default" else n_trees // 2 + 1
        self.b_strategy = b_strategy
        self.seed = seed

        self.data = ProgressiveDataStorage()

        self.forest = APForest(
            data=self.data,
            n_trees=n_trees,
            leaf_max_size=leaf_max_size,
            b_strategy=b_strategy,
            seed=seed,
        )
        self.cluster_handler = ClusterHandler(data=self.data, threshold=self.threshold)

    def update(self, batch: np.ndarray) -> ClusterUpdateEvent:
        start_idx = self.data.append(batch)

        self.forest.insert(start_idx)
        split_events = self.forest.split()
        all_leaf_nodes = self.forest.get_all_leaf_nodes()
        forest_id2node = self.forest.get_id_to_node_mappings()

        mc_split_events = self.cluster_handler.handle_split(
            start_idx, all_leaf_nodes, split_events
        )
        mc_merge_events, mc_creation_events = self.cluster_handler.handle_insertion(
            start_idx, forest_id2node
        )
        return ClusterUpdateEvent(
            split_events=mc_split_events,
            merge_events=mc_merge_events,
            creation_events=mc_creation_events,
        )

    def get_micro_clusters(self) -> list[MicroCluster]:
        return self.cluster_handler.micro_clusters
