from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prodr.ensemble.components import Node, MicroCluster


@dataclass
class InsertionEvent:
    """
    Event representing the insertion of a data point into a leaf node.
    """

    data_index: int
    node: Node


@dataclass
class NodeSplitEvent:
    """
    Event representing the split of a parent node into two child nodes.
    """

    parent_node: Node
    left_child: Node
    right_child: Node


@dataclass
class MicroClusterSplitEvent:
    """
    Event representing the inheritance of a micro-cluster by a new micro-cluster after a split.
    """

    parent_micro_cluster: MicroCluster
    child_micro_clusters: list[MicroCluster]
    inherit_micro_cluster: MicroCluster


@dataclass
class MicroClusterMergeEvent:
    """
    Event representing the merging of multiple micro-clusters into a single micro-cluster.
    """

    merged_micro_clusters: list[MicroCluster]
    head_micro_cluster: MicroCluster


@dataclass
class MicroClusterCreationEvent:
    """
    Event representing the creation of a new micro-cluster.
    """

    created_micro_cluster: MicroCluster


@dataclass
class ClusterUpdateEvent:
    """
    Event representing updates to clusters, including splits, merges, and creations.
    """

    split_events: list[MicroClusterSplitEvent]
    merge_events: list[MicroClusterMergeEvent]
    creation_events: list[MicroClusterCreationEvent]
