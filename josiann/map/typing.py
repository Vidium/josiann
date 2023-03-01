# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import Protocol
from typing import Iterator
from typing import TYPE_CHECKING

from josiann.storage.parameters import SAParameters

if TYPE_CHECKING:
    import josiann.typing as jot


# ====================================================
# code
class Execution(Protocol):
    def __call__(
        self,
        params: SAParameters,
        x: npt.NDArray[jot.DType],
        costs: npt.NDArray[np.float64],
        current_n: int,
        last_ns: npt.NDArray[np.int64],
        iteration: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[tuple[npt.NDArray[jot.DType], float, bool, int]]:
        ...
