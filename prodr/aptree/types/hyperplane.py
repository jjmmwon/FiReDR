from numbers import Number

from dataclasses import dataclass

from .vector import Vector


@dataclass(frozen=True)
class Hyperplane:
    """
    Represents a hyperplane in n-dimensional space.
    Attributes:
        normal (Vector): The normal vector perpendicular to the hyperplane.
        offset (Number): The offset distance from the origin along the normal vector.
    """

    normal: Vector
    offset: Number
