from __future__ import annotations
from typing import Optional

from dataclasses import dataclass

import numpy as np


from prodr.aptree.types import Hyperplane


@dataclass
class Node:
    """
    Represents a node in an adaptive partitioning tree.
    """

    data_indices: list[int]
    depth: int

    parent: Optional[Node] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    is_leaf: bool = True
    hyperplane: Optional[Hyperplane] = None

    def __post_init__(self) -> None:
        if self.is_leaf:
            self.left = None
            self.right = None
            self.hyperplane = None

    def is_root(self) -> bool:
        return self.parent is None
