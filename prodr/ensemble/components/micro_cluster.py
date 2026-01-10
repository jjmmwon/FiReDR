from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class MicroCluster:
    """
    A micro-cluster representing a small cluster of data points.

    Attributes:
        center (np.ndarray): The center of the micro-cluster.
        radius (float): The radius of the micro-cluster.
        weight (int): The number of data points in the micro-cluster.
    """

    data_indices: np.ndarray
    inner_structure: sp.csr_matrix
