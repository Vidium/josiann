# ====================================================
# imports
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.compute import updated_mean


# ====================================================
# code
def get_mean_cost(
    fun: jot.FUN_TYPE[...],
    x: npt.NDArray[jot.DType],
    _n: int,
    args: tuple[Any, ...],
    previous_evaluations: tuple[int, float],
) -> float:
    """
    Get the mean of <n> function evaluations for vector of values <x>.

    Args:
        fun: a function to evaluate.
        x: a vector of values.
        _n: the number of evaluations to compute.
        args: arguments to be passed to <fun>.
        previous_evaluations: previously computed function evaluations at position x: number of last function
            evaluations and obtained mean.

    Returns:
        The mean of function evaluations at x.
    """
    last_n, last_mean = previous_evaluations
    return updated_mean(
        last_n,
        last_mean,
        np.array([max(0.0, fun(x, *args)) for _ in range(_n - last_n)]),
    )
