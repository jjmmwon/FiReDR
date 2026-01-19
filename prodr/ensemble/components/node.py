from __future__ import annotations
from typing import Optional

from dataclasses import dataclass

from .hyperplane import Hyperplane


@dataclass
class Node:
    """
    Represents a node in an adaptive partitioning tree.
    """

    indices: list[int]
    depth: int
    is_leaf: bool = True

    parent: Optional[Node] = None

    _left: Optional[Node] = None
    _right: Optional[Node] = None
    _hyperplane: Optional[Hyperplane] = None

    def is_root(self) -> bool:
        return self.parent is None

    @property
    def left(self) -> Node:
        if self._left is None:
            raise RuntimeError("Leaf node has no left child.")
        return self._left

    @property
    def right(self) -> Node:
        if self._right is None:
            raise RuntimeError("Leaf node has no right child.")
        return self._right

    @property
    def hyperplane(self) -> Hyperplane:
        if self._hyperplane is None:
            raise RuntimeError("Leaf node has no hyperplane.")
        return self._hyperplane

    @left.setter
    def left(self, value: Node) -> None:
        self._left = value

    @right.setter
    def right(self, value: Node) -> None:
        self._right = value

    @hyperplane.setter
    def hyperplane(self, value: Hyperplane) -> None:
        self._hyperplane = value
