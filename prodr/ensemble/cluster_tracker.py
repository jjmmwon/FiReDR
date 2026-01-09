from prodr.components import MicroCluster, Node


class ClusterTracker:
    def __init__(self):
        self.micro_clusters: list[MicroCluster] = []
        self.idx_to_cluster: list[int] = []

    def initialization(self, E: list[list[Node]]):
        pass

    def update(self):
        pass

    def split_clusters(self):
        pass

    def merge_clusters(self):
        pass
