from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from prodr.ensemble.components import Node


class InsertionEvent(TypedDict):
    """
    Event representing the insertion of a data point into a leaf node.
    """

    data_index: int
    node: Node


class SplitEvent(TypedDict):
    """
    Event representing the split of a parent node into two child nodes.
    """

    parent_node: Node
    left_child: Node
    right_child: Node
