from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from prodr.ensemble.validators import check_feature_dim, check_dtype


@dataclass
class ProgressiveDataStorage:
    """
    A data storage class that allows
    appending new data in chunks and retrieving the full dataset.
    Attributes:
        n_features (int | None): Number of features in the dataset.
        dtype (np.dtype | None): Data type of the dataset.
    """

    n_features: int | None = None
    dtype: np.dtype | None = None
    _X: np.ndarray | None = None
    # _X_dirty: bool = False
    _n_samples: int = 0

    # _chunks: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _starts: list[int] = field(default_factory=list, init=False, repr=False)

    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        return self._X.shape[0] if self._X is not None else 0

    def append(self, batch: np.ndarray) -> int:
        if self.n_features is None:
            self.n_features = batch.shape[1]
            self.dtype = batch.dtype
        else:
            check_feature_dim(batch, self.n_features)
            check_dtype(batch.dtype, self.dtype)

        start = self._n_samples
        # self._chunks.append(batch)
        self._starts.append(self._n_samples)
        # self._X_dirty = True
        self._n_samples += batch.shape[0]
        self._X = np.vstack([self._X, batch]) if self._X is not None else batch

        return start

    # def get_data(self) -> np.ndarray:
    #     if self._X is None and not self._chunks:
    #         raise ValueError("No data available.")

    #     if self._X is None:
    #         self._X = np.vstack(self._chunks)
    #     elif self._X_dirty:
    #         self._X = np.vstack([self._X] + self._chunks)
    #         self._chunks = []
    #         self._X_dirty = False

    #     return self._X

    def __getitem__(self, idx: int | slice | np.ndarray | Sequence[int]) -> np.ndarray:
        if self._X is None:
            raise ValueError("No data available.")
        X = self._X
        return X[idx]
