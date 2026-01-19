import scipy.sparse as sp

from prodr.ensemble.components import MicroCluster


def merge_micro_clusters(
    micro_clusters: list[MicroCluster],
) -> tuple[MicroCluster, MicroCluster]:
    micro_clusters = sorted(micro_clusters, key=lambda mc: mc.size, reverse=True)

    head_cluster = micro_clusters[0]

    for mc in micro_clusters[1:]:
        head_cluster.indices += mc.indices

    matrices = [mc.cooccurrence_count for mc in micro_clusters]

    head_cluster.cooccurrence_count = sp.block_diag(matrices, format="csr")  # type: ignore

    return (
        MicroCluster(
            indices=head_cluster.indices,
            cooccurrence_count=head_cluster.cooccurrence_count,
            head=head_cluster.head,
        ),
        head_cluster,
    )
