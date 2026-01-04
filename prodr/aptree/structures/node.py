from __future__ import annotations
from typing import Optional

from dataclasses import dataclass


from prodr.aptree.types import Hyperplane, Instance


@dataclass
class Node:
    """
    Represents a node in an adaptive partitioning tree.
    """

    parent: Optional[Node]
    left: Optional[Node]
    right: Optional[Node]
    is_leaf: bool
    hyperplane: Optional[Hyperplane]
    data: Optional[list[Instance]]
    depth: int
