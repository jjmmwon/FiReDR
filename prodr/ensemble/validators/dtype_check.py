def check_dtype(dtype1, dtype2) -> None:
    """
    Check if two data types are the same.
    Args:
        dtype1 (np.dtype): First data type.
        dtype2 (np.dtype): Second data type.
    Raises:
        ValueError: If the data types do not match.
    """
    if dtype1 != dtype2:
        raise ValueError(f"Dtype mismatch: expected {dtype1}, got {dtype2}")
