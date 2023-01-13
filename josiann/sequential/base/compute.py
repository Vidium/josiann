# coding: utf-8
# Created on 12/01/2023 11:34
# Author : matteo
from __future__ import annotations

import numpy.typing as npt
from typing import Any

import josiann.typing as jot


# ====================================================
# imports

# ====================================================
# code
def get_mean_cost(
    fun: jot.FUN_TYPE,
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
    return float(
        last_mean * last_n / _n
        + sum([max(0.0, fun(x, *args)) for _ in range(_n - last_n)]) / _n
    )
