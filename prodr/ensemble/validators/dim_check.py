import numpy as np


def check_feature_dim(data: np.ndarray, expected_dim: int) -> None:
    """
    Check if the data has the expected dimensionality.
    Args:
        data (np.ndarray): Input data array.
        expected_dim (int): Expected number of dimensions.
    Raises:
        ValueError: If the data does not have the expected dimensionality.
    """
    if data.shape[1] != expected_dim:
        raise ValueError(
            f"Data dimensionality mismatch: expected {expected_dim}, got {data.shape[1]}"
        )
