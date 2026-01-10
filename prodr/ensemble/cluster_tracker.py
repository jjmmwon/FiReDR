import numpy as np

from .components import MicroCluster, Node
from .utils import generate_collision_matrix, extract_micro_clusters


class ClusterTracker:
    def __init__(self):
        self.micro_clusters: list[MicroCluster] = []
        self.centers: list[int] = []

    def initialization(
        self, E: list[list[Node]], n_samples: int, threshold: int | None = None
    ) -> None:
        threshold = threshold if threshold is not None else len(E) // 2 + 1

        collision_matrix = generate_collision_matrix(E, n_samples)
        members, micro_clusters = extract_micro_clusters(collision_matrix, threshold)

        self.centers = [member[0] for member in members]
        self.micro_clusters = [
            MicroCluster(
                data_indices=member,
                inner_structure=micro_cluster,
            )
            for member, micro_cluster in zip(members, micro_clusters)
        ]

    def update(self):
        pass

    def split_clusters(self):
        pass

    def merge_clusters(self):
        pass
