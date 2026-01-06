from dataclasses import dataclass

import numpy as np

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
    offset: float

    def evaluate(self, x: Vector) -> float:
        """
        Evaluate the hyperplane equation for a given point.
        Args:
            x (Vector): The point to evaluate.
        Returns:
            Number: The result of the hyperplane equation.
        """
        return np.dot(self.normal, x) + self.offset
