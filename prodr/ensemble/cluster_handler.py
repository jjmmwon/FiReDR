import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .types import (
    NodeSplitEvent,
    MicroClusterSplitEvent,
    MicroClusterMergeEvent,
    MicroClusterCreationEvent,
)
from .components import MicroCluster, Node, ProgressiveDataStorage
from .utils import (
    generate_cooccurr_acc_mtx,
    split_micro_cluster,
    count_mcs_new_data_cooccurrence,
    update_micro_clusters_with_new_data,
)


class ClusterHandler:
    """
    Handler for managing micro-clusters based on ensemble tree events.
    Attributes:
        micro_clusters (list[MicroCluster]): List of current micro-clusters.
        id_to_mc (list[MicroCluster]): Mapping from data point IDs to their corresponding micro
        clusters.
    """

    def __init__(
        self,
        *,
        data: ProgressiveDataStorage,
        threshold: int,
    ) -> None:
        self.data = data
        self.threshold = threshold

        self.micro_clusters: list[MicroCluster] = []
        self.id_to_mc: dict[int, MicroCluster] = {}
        self.mcid_to_mc: dict[int, MicroCluster] = {}

        self._initialized = False
        self._initialization_phase = False

    def _ensure_initialized(self, all_leaf_nodes: list[list[Node]]) -> bool:
        if self._initialized:
            return True
        elif len(all_leaf_nodes[0]) > 8:
            self._initialization(all_leaf_nodes)
            self._initialized = True
            self._initialization_phase = True
            return True
        else:
            return False

    def _initialization(self, all_leaf_nodes: list[list[Node]]) -> None:
        n_samples = self.data.size

        cooccurr_cnt_list = [
            [node.indices for node in tree_leaf_nodes]
            for tree_leaf_nodes in all_leaf_nodes
        ]
        cooccurr_cnt_mtx = generate_cooccurr_acc_mtx(cooccurr_cnt_list, n_samples)

        init_mc = MicroCluster(
            indices=list(range(n_samples)),
            cooccurrence_count=cooccurr_cnt_mtx,
            head=0,
        )

        micro_clusters, _ = split_micro_cluster(
            init_mc,
            self.threshold,
        )

        self.micro_clusters = micro_clusters
        self._initialize_id_to_mc_mapping()

    def _initialize_id_to_mc_mapping(self) -> None:
        for mc in self.micro_clusters:
            for i in mc.indices:
                self.id_to_mc[i] = mc

    def handle_split(
        self,
        start_idx: int,
        all_leaf_nodes: list[list[Node]],
        split_events: list[list[NodeSplitEvent]],
    ) -> list[MicroClusterSplitEvent]:
        if not self._ensure_initialized(all_leaf_nodes):
            return []
        mc_split_events: list[MicroClusterSplitEvent] = []

        dirty_mcs: set[MicroCluster] = set()

        for tree_events in split_events:
            for event in tree_events:
                left_ids = [i for i in event.left_child.indices if i < start_idx]
                right_ids = [i for i in event.right_child.indices if i < start_idx]

                per_mc_left: dict[MicroCluster, list[int]] = {}
                per_mc_right: dict[MicroCluster, list[int]] = {}

                for lid in left_ids:
                    mc = self.id_to_mc[lid]
                    mc.get_local_idx(lid)
                    per_mc_left.setdefault(mc, []).append(lid)

                for rid in right_ids:
                    mc = self.id_to_mc[rid]
                    mc.get_local_idx(rid)
                    per_mc_right.setdefault(mc, []).append(rid)

                common_mcs = set(per_mc_left.keys()).intersection(per_mc_right.keys())

                for mc in common_mcs:
                    lids = per_mc_left[mc]
                    rids = per_mc_right[mc]

                    rows = [lid for lid in lids for _ in rids]
                    cols = [rid for _ in lids for rid in rids]
                    counts = [-1] * len(rows)

                    mc.update_cooccurrence_count(
                        rows,
                        cols,
                        counts,
                    )

                    if mc.is_dirty(self.threshold):
                        dirty_mcs.add(mc)

        for mc in dirty_mcs:
            new_mcs, inherit_mc_label = split_micro_cluster(mc, self.threshold)

            self.micro_clusters.remove(mc)
            self.micro_clusters.extend(new_mcs)

            mc_split_events.append(
                MicroClusterSplitEvent(
                    parent_micro_cluster=mc,
                    child_micro_clusters=new_mcs,
                    inherit_micro_cluster=new_mcs[inherit_mc_label],
                )
            )

            for new_mc in new_mcs:
                for i in new_mc.indices:
                    self.id_to_mc[i] = new_mc

        return mc_split_events

    def handle_insertion(
        self,
        start_idx: int,
        forest_id2node: list[list[Node]],
    ) -> tuple[
        list[MicroClusterMergeEvent],
        list[MicroClusterCreationEvent],
    ]:
        if not self._initialized or self._initialization_phase:
            self._initialization_phase = False
            return [], []

        end_idx = self.data.size - 1

        coocc_mtx, neighbors_of_new = count_mcs_new_data_cooccurrence(
            micro_clusters=self.micro_clusters,
            new_data_idx_range=(start_idx, end_idx),
            forest_id2node=forest_id2node,
            threshold=self.threshold,
            id_to_mc=self.id_to_mc,
        )

        merge_events, creation_events, created_mcs, removed_mcs = (
            update_micro_clusters_with_new_data(
                coocurrence_matrix=coocc_mtx,
                micro_clusters=self.micro_clusters,
                start_idx=start_idx,
                neighbors_of_new_data=neighbors_of_new,
            )
        )

        if removed_mcs:
            self.micro_clusters = [
                mc for mc in self.micro_clusters if mc not in removed_mcs
            ]

        if created_mcs:
            self.micro_clusters.extend(created_mcs)
            for mc in created_mcs:
                for i in mc.indices:
                    self.id_to_mc[i] = mc

        return merge_events, creation_events

        # # TODO: mc_indices 중에서 제일 큰걸 찾아서 그것끼리 병합
        # # TODO: 그렇게 나온 대장에게 data_indices를 한꺼번에 추가 (data_indices)끼리 먼저합치고 합치는게 나을수도?

        # # TODO: UMAP용 로그 남기기

        # # TODO: data_indices 중에 누구랑도 연결안되어서 혼자남은애들을 각각 MC로 추가
        # # TODO: UMAP용 로그 남기기
        # # TODO: id_to_mc 갱신
