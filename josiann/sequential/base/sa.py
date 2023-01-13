# coding: utf-8
# Created on 03/12/2022 13:40
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import time
import numpy as np
from tqdm.autonotebook import tqdm
from tqdm.autonotebook import trange

import numpy.typing as npt
from typing import Any
from typing import Sequence

from josiann.map.linear import linear_execution
from josiann.run import run_simulated_annealing
from josiann.storage.result import Result
from josiann.storage.trace import OneTrace
from josiann.sequential.base.intialize import initialize_sa
from josiann.moves.base import Move
from josiann.moves.sequential import RandomStep

import josiann.typing as jot


# ====================================================
# code
def sa(
    fun: jot.FUN_TYPE,
    x0: npt.NDArray[Any],
    args: tuple[Any, ...] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    moves: Move
    | Sequence[Move]
    | Sequence[tuple[float, Move]] = ((0.8, RandomStep(0.05)), (0.2, RandomStep(0.5))),
    nb_walkers: int = 1,
    max_iter: int = 200,
    max_measures: int = 20,
    final_acceptance_probability: float = 1e-300,
    epsilon: float = 0.01,
    T_0: float = 5.0,
    tol: float = 1e-3,
    backup: bool = False,
    seed: int = int(time.time()),
    verbose: bool = True,
    suppress_warnings: bool = False,
    detect_convergence: bool = True,
    window_size: int | None = None,
    dtype: jot.DType = np.float64,  # type: ignore[assignment]
) -> Result:
    """
    Simulated Annealing for minimizing noisy cost functions.

    Args:
        fun: a <d> dimensional (noisy) function to minimize.
        x0: a <d> dimensional vector of initial values.
        args: an optional sequence of arguments to pass to the function to minimize.
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        moves: either
                    - a single josiann.Move object
                    - a sequence of josiann.Move objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.Move)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        backup: use Backup for storing previously computed function evaluations and reusing them when returning to
            the same position vector ? (Only available when using SetStep moves).
        seed: a seed for the random generator.
        verbose: print progress bar ? (default True)
        suppress_warnings: remove warnings ? (default False)
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.
        dtype: the data type for the values stored in the Trace. (default np.float64)

    Returns:
        A Result object.
    """
    params = initialize_sa(
        args,
        x0,
        nb_walkers,
        max_iter,
        max_measures,
        final_acceptance_probability,
        epsilon,
        T_0,
        tol,
        moves,
        bounds,
        fun,
        backup,
        suppress_warnings,
        detect_convergence,
        window_size,
        seed,
        dtype,
    )

    x = params.base.x
    costs = params.costs
    last_ns = params.last_ns

    # initialize the trace history keeper
    trace = OneTrace(
        nb_iterations=params.base.max_iter,
        nb_walkers=x.shape[0],
        nb_dimensions=x.shape[1],
        run_parameters=params,
        initial_position=x,
        initial_cost=np.array(costs),
    )

    progress_bar: range | tqdm[int]
    if verbose:
        progress_bar = trange(params.base.max_iter, unit="iteration")
    else:
        progress_bar = range(params.base.max_iter)

    # run the SA algorithm
    return run_simulated_annealing(
        np.array(costs),
        linear_execution,
        np.array(last_ns),
        nb_walkers,
        params,
        progress_bar,
        trace,
        x,
    )
