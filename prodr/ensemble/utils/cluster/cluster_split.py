from scipy.sparse.csgraph import connected_components

from prodr.ensemble.components import MicroCluster
from .cluster_generation import generate_micro_clusters


def split_micro_cluster(
    micro_cluster: MicroCluster, threshold: int
) -> tuple[list[MicroCluster], int]:
    members, cooccurr_mtx = (
        micro_cluster.indices,
        micro_cluster.cooccurrence_count,
    )
    # Create a binary adjacency matrix based on the threshold
    adjacency_matrix = cooccurr_mtx >= threshold

    filtered_inner_structure = cooccurr_mtx.multiply(adjacency_matrix)
    filtered_inner_structure.eliminate_zeros()

    # Find connected components in the graph represented by the adjacency matrix
    n_components, labels = connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )

    head_global_idx = micro_cluster.head

    return (
        generate_micro_clusters(
            n_components,
            labels,
            filtered_inner_structure,
            members,
            head_global_idx,
        ),
        labels[micro_cluster.get_local_idx(head_global_idx)],
    )
