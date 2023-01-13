# coding: utf-8
# Created on 04/12/2022 10:46
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

from typing import Union
from typing import TypeVar
from typing import Callable
from typing_extensions import Concatenate
from typing_extensions import ParamSpec
from numpy.typing import NDArray


# ====================================================
# code
DType = Union[np.float64, np.int64]
DT_ARR = TypeVar("DT_ARR", bound=DType)

_P = ParamSpec("_P")

FUN_TYPE = Callable[Concatenate[NDArray[np.float64], _P], float]  # type: ignore[valid-type, misc]
VECT_FUN_TYPE = Callable[Concatenate[NDArray[np.float64], _P], list[float]]  # type: ignore[valid-type, misc]

# SA_UPDATE = tuple[Union[NDArray[np.float64], float], float, bool, int]
