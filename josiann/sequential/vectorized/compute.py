# coding: utf-8
# Created on 03/12/2022 17:16
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any

import josiann.typing as jot


# ====================================================
# code
def get_evaluation_vectorized_mean_cost(
    fun: jot.VECT_FUN_TYPE,
    x: npt.NDArray[np.float64 | np.int64],
    _n: int,
    args: tuple[Any, ...],
    previous_evaluations: list[tuple[int, float]],
) -> list[float]:
    """
    Same as 'get_mean_cost' but <fun> is a vectorized function and costs are computed for all walkers at once.

    Args:
        fun: a vectorized function to evaluate.
        x: a matrix of position vectors of shape (nb_walkers, d).
        _n: the number of evaluations to compute.
        args: arguments to be passed to <fun>.
        previous_evaluations: list of previously computed function evaluations at position x: number of last function
            evaluations and obtained means for each walker position.

    Returns:
        The mean of function evaluations at x.
    """
    evaluations = [0.0 for _ in range(len(x))]

    for walker_index, walker_position in enumerate(x):
        last_n, last_mean = previous_evaluations[walker_index]
        remaining_n = _n - last_n
        if remaining_n:
            evaluations[walker_index] = (
                last_mean * last_n / _n
                + sum(fun(np.tile(walker_position, (remaining_n, 1)), *args)) / _n
            )

        else:
            evaluations[walker_index] = last_mean

    return evaluations


def get_walker_vectorized_mean_cost(
    fun: jot.VECT_FUN_TYPE,
    x: npt.NDArray[np.float64 | np.int64],
    _n: int,
    args: tuple[Any, ...],
    previous_evaluations: list[tuple[int, float]],
    vectorized_skip_marker: Any,
) -> list[float]:
    """
    Same as 'get_mean_cost' but <fun> is a vectorized function and costs are computed for all walkers at once but
        sequentially on function evaluations.

    Args:
        fun: a vectorized function to evaluate.
        x: a matrix of position vectors of shape (nb_walkers, d).
        _n: the number of evaluations to compute.
        args: arguments to be passed to <fun>.
        previous_evaluations: list of previously computed function evaluations at position x: number of last function
            evaluations and obtained means for each walker position.
        vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
            evaluation for a particular position vector can be skipped.

    Returns:
        The mean of function evaluations at x.
    """
    zipped_last = zip(
        *[previous_evaluations[walker_index] for walker_index, _ in enumerate(x)]
    )
    last_n = list(next(zipped_last))
    remaining_n = [_n - ln for ln in last_n]
    last_mean = list(next(zipped_last))

    if max(remaining_n):
        costs = np.zeros(len(x))

        for eval_index in range(max(remaining_n)):
            eval_vector = np.array(
                [
                    walker_position
                    if eval_index < remaining_n[walker_index]
                    else vectorized_skip_marker
                    for walker_index, walker_position in enumerate(x)
                ]
            )

            res = np.array(fun(eval_vector, *args))

            for walker_index, _ in enumerate(res):
                if eval_index >= remaining_n[walker_index]:
                    res[walker_index] = 0.0

            costs += res

        return list((np.array(last_mean) * last_n + costs) / _n)

    return last_mean
