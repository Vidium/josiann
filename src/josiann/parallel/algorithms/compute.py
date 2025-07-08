# ====================================================
# imports
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.compute import updated_mean
from josiann.parallel.arguments import ParallelArgument


# ====================================================
# code
def get_vectorized_mean_cost(
    fun: jot.PARALLEL_FUN_TYPE[...],
    x: npt.NDArray[np.float64 | np.int64],
    n: int,
    parallel_args: tuple[npt.NDArray[Any], ...],
    args: tuple[Any, ...],
    previous_evaluations: list[tuple[int, float]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Get the mean of <n> function evaluations for vectors of values <x>.

    Args:
        fun: a vectorized function to evaluate.
        x: a matrix of position vectors of shape (nb_parallel_problems, d).
        n: the number of evaluations to compute.
        parallel_args: an optional sequence of arrays (of size equal to the number of parallel problems) of arguments
            to pass to the vectorized function to minimize. Parallel arguments are passed before other arguments.
        args: arguments to be passed to <fun>.
        previous_evaluations: list of previously computed function evaluations at position x: number of last function
            evaluations and obtained means for each walker position.

    Returns:
        The mean of function evaluations at x.
    """
    remaining_ns = n - np.array([last_n for last_n, _ in previous_evaluations])

    arguments = ParallelArgument(positions=x, nb_evaluations=remaining_ns, args=parallel_args)

    fun(arguments, *args)

    return np.array(
        [
            updated_mean(last_n, last_mean, res)
            for res, (last_n, last_mean) in zip(arguments.result_iter(), previous_evaluations)
        ]
    )
