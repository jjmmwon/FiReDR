import scipy.sparse as sp


from prodr.ensemble.components import MicroCluster, Node
from prodr.ensemble.types import MicroClusterCreationEvent, MicroClusterMergeEvent
from prodr.ensemble.utils import count_cooccurrence, merge_micro_clusters


def count_mcs_new_data_cooccurrence(
    micro_clusters: list[MicroCluster],
    new_data_idx_range: tuple[int, int],
    forest_id2node: list[list[Node]],
    threshold: int,
    id_to_mc: dict[int, MicroCluster],
) -> tuple[sp.csr_array, dict[int, list[tuple[int, int]]]]:
    """
    Count cooccurrence between existing micro-clusters and new data points.

    Args:
        micro_clusters (list[MicroCluster]): List of existing micro-clusters.
        range(start_idx, end_idx + 1) (list[int]): List of new data point indices.

    Returns:
        sp.csr_array: Cooccurrence count matrix of shape
            (n_micro_clusters, n_new_data_points).
    """
    start_idx, end_idx = new_data_idx_range
    n_mcs = len(micro_clusters)
    n_new = end_idx - start_idx + 1
    n_total = n_mcs + n_new

    neighbors_of_new: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(start_idx, end_idx + 1)
    }
    mc2mcidx = {mc: idx for idx, mc in enumerate(micro_clusters)}

    rows = []
    cols = []
    counts = []

    for new_idx in range(start_idx, end_idx + 1):
        assigned_nodes = [tree_id2node[new_idx] for tree_id2node in forest_id2node]

        neighbors = count_cooccurrence(assigned_nodes, threshold)
        neighbors.pop(new_idx, None)

        i = n_mcs + (new_idx - start_idx)

        for neighbor_idx, count in neighbors.items():
            if neighbor_idx >= start_idx:
                j = n_mcs + (neighbor_idx - start_idx)

            else:
                mc = id_to_mc[neighbor_idx]
                j = mc2mcidx[mc]
                neighbors_of_new[new_idx].append((neighbor_idx, count))

            rows.append(i)
            cols.append(j)
            counts.append(count)

    coocc = sp.coo_array(
        (counts, (rows, cols)), shape=(n_total, n_total), dtype=int
    ).tocsr()
    coocc = coocc + coocc.T

    return coocc, neighbors_of_new


def update_micro_clusters_with_new_data(
    coocurrence_matrix: sp.csr_array,
    micro_clusters: list[MicroCluster],
    start_idx: int,
    neighbors_of_new_data: dict[int, list[tuple[int, int]]],
) -> tuple[
    list[MicroClusterMergeEvent],
    list[MicroClusterCreationEvent],
    list[MicroCluster],
    set[MicroCluster],
]:
    mc_merge_events: list[MicroClusterMergeEvent] = []
    mc_creation_events: list[MicroClusterCreationEvent] = []
    created_mcs = []
    removed_mcs: set[MicroCluster] = set()

    n_mcs = len(micro_clusters)

    _, labels = sp.csgraph.connected_components(
        coocurrence_matrix, directed=False, return_labels=True
    )

    indices_by_label: dict[int, list[int]] = {}

    for idx, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(idx)

    for label, indices in indices_by_label.items():
        mcs_to_merge: list[MicroCluster] = []

        new_data_global_ids = []
        new_data_local_ids = []
        cooccurr_cnts_to_update = []

        for idx in indices:
            if idx < n_mcs:
                mc = micro_clusters[idx]
                mcs_to_merge.append(mc)
            else:
                new_data_global_id = start_idx + (idx - n_mcs)
                new_data_global_ids.append(new_data_global_id)
                new_data_local_ids.append(idx)

                cooccurr_cnts_to_update.extend(
                    [
                        (new_data_global_id, *neighbor_and_count)
                        for neighbor_and_count in neighbors_of_new_data[
                            new_data_global_id
                        ]
                    ]
                )
        if not new_data_global_ids:
            continue

        new_mc_cooccurr_mtx = coocurrence_matrix[new_data_local_ids][
            :, new_data_local_ids
        ]

        new_mc = MicroCluster(
            indices=new_data_global_ids,
            cooccurrence_count=new_mc_cooccurr_mtx,
            head=new_data_global_ids[0],
        )

        if mcs_to_merge:
            merged_mc, head_mc = merge_micro_clusters(mcs_to_merge + [new_mc])
            merged_mc.update_cooccurrence_count(
                [update[0] for update in cooccurr_cnts_to_update],
                [update[1] for update in cooccurr_cnts_to_update],
                [update[2] for update in cooccurr_cnts_to_update],
            )
            mc_merge_events.append(
                MicroClusterMergeEvent(
                    merged_micro_clusters=mcs_to_merge + [new_mc],
                    head_micro_cluster=head_mc,
                )
            )
            removed_mcs.update(mcs_to_merge)
            created_mcs.append(merged_mc)

        else:
            mc_creation_events.append(
                MicroClusterCreationEvent(created_micro_cluster=new_mc)
            )
            created_mcs.append(new_mc)

    return mc_merge_events, mc_creation_events, created_mcs, removed_mcs
