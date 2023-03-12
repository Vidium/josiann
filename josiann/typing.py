# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy

from typing import Union
from typing import TypeVar
from typing import Callable
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from typing_extensions import Concatenate
    from typing_extensions import ParamSpec
    from josiann.parallel import ParallelArgument


# ====================================================
# code
DType: TypeAlias = Union[numpy.float64, numpy.int64]
DT_ARR = TypeVar("DT_ARR", bound=DType)


if TYPE_CHECKING:
    # waiting for mypy to allow Concatenate[NDArray[DType], ...] as we don't care about the other parameters,
    # for now I have to use a ParamSpec
    # (https://github.com/python/mypy/issues/14656)
    _P = ParamSpec("_P")

    FUN_TYPE: TypeAlias = Callable[Concatenate[NDArray[DType], _P], float]
    VECT_FUN_TYPE: TypeAlias = Callable[Concatenate[NDArray[DType], _P], list[float]]
    PARALLEL_FUN_TYPE: TypeAlias = Callable[Concatenate[ParallelArgument, _P], None]
