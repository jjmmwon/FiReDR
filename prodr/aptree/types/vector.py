from typing import Sequence
from numbers import Number

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.number] | Sequence[np.number | Number]
