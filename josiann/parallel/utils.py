# coding: utf-8
# Created on 16/06/2022 11:43
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

from typing import Callable, Sequence


# ====================================================
# code
def get_vectorized_mean_cost(fun: Callable,
                             x: np.ndarray,
                             _n: int,
                             converged: np.ndarray,
                             parallel_args: Sequence[np.ndarray] | None,
                             args: tuple,
                             previous_evaluations: list[tuple[int, float]]) -> list[float]:
    """
    Get the mean of <n> function evaluations for vectors of values <x>.

    Args:
        fun: a vectorized function to evaluate.
        x: a matrix of position vectors of shape (nb_parallel_problems, d).
        _n: the number of evaluations to compute.
        converged: a vector indicating which walkers have already converged.
        parallel_args: an optional sequence of arrays (of size equal to the number of parallel problems) of arguments
            to pass to the vectorized function to minimize. Parallel arguments are passed before other arguments.
        args: arguments to be passed to <fun>.
        previous_evaluations: list of previously computed function evaluations at position x: number of last function
            evaluations and obtained means for each walker position.

    Returns:
        The mean of function evaluations at x.
    """
    # TODO : switch to median

    remaining_ns = _n - np.array([last_n for last_n, _ in previous_evaluations])

    all_x = np.repeat(x, remaining_ns, axis=0)

    all_parallel_args = () if parallel_args is None else [np.repeat(arg[~converged], remaining_ns)
                                                          for arg in parallel_args]

    all_evaluations = fun(all_x, converged, *all_parallel_args, *args)

    evaluations = []
    evaluation_index_start = 0
    for last_n, last_mean in previous_evaluations:
        if _n - last_n:
            evaluation_index_stop = evaluation_index_start + _n - last_n
            evaluations.append(last_mean * last_n / _n +
                               sum(all_evaluations[evaluation_index_start:evaluation_index_stop]) / _n)

            evaluation_index_start = evaluation_index_stop

        else:
            evaluations.append(last_mean)

    return evaluations
