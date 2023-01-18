# coding: utf-8
# Created on 15/06/2022 17:14
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import time
import traceback
import numpy as np
from tqdm.autonotebook import tqdm
from tqdm.autonotebook import trange

import numpy.typing as npt
from typing import Any
from typing import Sequence

from josiann.compute import n, T, sigma
from josiann.parallel.mappers import vectorized_execution
from josiann.storage.parallel.parameters import initialize_sa
from josiann.storage.parallel.trace import ParallelTrace
from josiann.storage.result import Result
from josiann.moves.parallel.base import ParallelMove
from josiann.moves.parallel.set import ParallelSetStep

import josiann.typing as jot


# ====================================================
# code
def psa(
    fun: jot.VECT_FUN_TYPE,
    x0: npt.NDArray[Any],
    parallel_args: Sequence[npt.NDArray[Any]] | None = None,
    args: tuple[Any, ...] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    moves: ParallelMove
    | Sequence[ParallelMove]
    | Sequence[tuple[float, ParallelMove]] = ParallelSetStep(
        np.tile(np.arange(-1, 1.1, 0.1), (2, 1))
    ),
    max_iter: int = 200,
    max_measures: int = 20,
    final_acceptance_probability: float = 1e-300,
    epsilon: float = 0.01,
    T_0: float = 5.0,
    tol: float = 1e-3,
    backup: bool = False,
    seed: int = int(time.time()),
    verbose: bool = True,
    leave_progress_bar: bool = True,
    suppress_warnings: bool = False,
    detect_convergence: bool = True,
    window_size: int | None = None,
    dtype: jot.DType = np.float64,  # type: ignore[assignment]
) -> Result:
    """
    Simulated Annealing for minimizing noisy cost functions in parallel. This is equivalent to running several
        Josiann.sa instances in parallel but all independent optimization problems are treated at once.

    Args:
        fun: a <d> dimensional (noisy) function to minimize.
        x0: a matrix of <d> dimensional vectors of initial values, one per parallel problem.
        parallel_args: an optional sequence of arrays (of size equal to the number of parallel problems) of arguments
            to pass to the vectorized function to minimize. Parallel arguments are passed before other arguments.
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
        leave_progress_bar: leave progress bar after the algorithm has completed (only works if verbose) ?
            (default True)
        suppress_warnings: remove warnings ? (default False)
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.
        dtype: the data type for the values stored in the Trace. (default np.float64)

    Returns:
        A Result object.
    """
    params = initialize_sa(
        parallel_args,
        args,
        x0,
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
    trace = ParallelTrace(
        nb_iterations=params.base.max_iter,
        nb_walkers=x.shape[0],
        nb_dimensions=x.shape[1],
        run_parameters=params,
        initial_position=x,
        initial_cost=np.array(costs),
    )

    progress_bar: range | tqdm[int]
    if verbose:
        progress_bar = trange(
            params.base.max_iter, unit="iteration", leave=leave_progress_bar
        )
    else:
        progress_bar = range(params.base.max_iter)

    # run the SA algorithm
    iteration = 0

    for iteration in progress_bar:
        temperature = T(iteration, params.base.T_0, params.base.alpha)
        current_n = n(iteration, params.base)
        current_sigma = sigma(
            iteration, params.base.T_0, params.base.alpha, params.base.epsilon
        )

        accepted = np.zeros(params.multi.nb_walkers, dtype=bool)
        explored = np.zeros((params.multi.nb_walkers, params.base.nb_dimensions))
        explored_costs = np.zeros(params.multi.nb_walkers)

        start = time.perf_counter()

        try:
            updates = vectorized_execution(
                params.fun,
                x.copy(),
                trace.positions.converged,
                costs=costs,
                current_n=current_n,
                last_ns=last_ns,
                parallel_args=params.base.parallel_args,
                args=params.base.args,
                list_moves=params.moves.list_moves,
                list_probabilities=params.moves.list_probabilities,
                temperature=temperature,
                positions=x.copy(),
                backup=params.backup,
            )

            for _x, _cost, _accepted, _walker_index in updates:
                if _accepted:
                    # update positions, costs, n
                    x[_walker_index] = _x
                    costs[_walker_index] = _cost
                    last_ns[_walker_index] = current_n

                accepted[_walker_index] = _accepted
                explored[_walker_index] = _x
                explored_costs[_walker_index] = _cost

        except Exception:
            message = (
                f"Unexpected failure while evaluating cost function : \n"
                f"{traceback.format_exc()}"
            )
            success = False
            break

        elapsed = time.perf_counter() - start

        trace.positions.store(
            iteration, x, np.array(costs), current_n, accepted, explored, explored_costs
        )
        trace.parameters.store(
            iteration, temperature, current_n, current_sigma, elapsed
        )

        if isinstance(progress_bar, tqdm):
            progress_bar.set_description(
                f"T: {temperature:.4f}"
                f"  n: {current_n}"
                f"  Converged: {np.sum(trace.positions.converged)}/{params.multi.nb_walkers}"
            )

        if np.all(trace.positions.converged):
            message, success = "Convergence tolerance reached.", True
            break

    else:
        message, success = "Requested number of iterations reached.", False

    trace.finalize(iteration)

    return Result(message, success, trace, params)
