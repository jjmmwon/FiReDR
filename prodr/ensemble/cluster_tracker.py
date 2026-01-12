import numpy as np

from .types import UpdateLog
from .components import MicroCluster, Node
from .utils import generate_collision_matrix, extract_micro_clusters


class ClusterTracker:
    def __init__(self):
        self.micro_clusters: list[MicroCluster] = []
        self.centers: list[int] = []

        self._initialized = False

    def _ensure_initialized(self, E: list[list[Node]]) -> None:
        if self._initialized:
            return

        if len(E[0]) > 8:
            self.initialization(E=E)
            self._initialized = True

    def initialization(self, E: list[list[Node]], threshold: int | None = None) -> None:
        threshold = threshold if threshold is not None else len(E) // 2 + 1
        n_samples = sum(len(node.data_indices) for node in E[0])

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

    def insert(self, E: list[list[Node]], update_logs: list[UpdateLog]):
        self._ensure_initialized(E)
        self.update(update_logs)

    def update(self, update_logs: list[UpdateLog]):
        insertion_log = [log.insertion_log for log in update_logs]
        split_log = [log.split_log for log in update_logs]
        self._update_insertion_events()
        pass

    def _update_insertion_events(self):
        pass

    def split_clusters(self):
        pass

    def merge_clusters(self):
        pass
