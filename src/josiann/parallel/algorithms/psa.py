from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm.autonotebook import tqdm, trange

import josiann.typing as jot
from josiann.compute import T, n, sigma
from josiann.parallel.algorithms.mappers import UpdateState, vectorized_execution
from josiann.parallel.moves.base import ParallelMove
from josiann.parallel.moves.discrete import ParallelSetStep
from josiann.storage.parallel.parameters import initialize_sa
from josiann.storage.parallel.trace import ParallelTrace
from josiann.storage.result import Result


def psa(
    fun: Callable[..., None],
    x0: npt.NDArray[Any],
    parallel_args: Sequence[npt.NDArray[Any]] | None = None,
    args: tuple[Any, ...] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    moves: ParallelMove | Sequence[ParallelMove] | Sequence[tuple[float, ParallelMove]] = ParallelSetStep(
        position_set=[list(np.arange(-1, 1.1, 0.1)), list(np.arange(-1, 1.1, 0.1))]
    ),
    max_iter: int = 200,
    max_measures: int = 20,
    alpha: float = 0.95,
    epsilon: float = 0.01,
    T_0: float = 5.0,
    tol: float = 1e-3,
    backup: bool = False,
    seed: int | None = None,
    verbose: bool = True,
    leave_progress_bar: bool = True,
    suppress_warnings: bool = False,
    detect_convergence: bool = True,
    window_size: int | None = None,
    dtype: jot.DType = np.float64,  # type: ignore[assignment]
) -> Result:
    """
    Simulated Annealing algorithm solving multiple problems in parallel.

    Args:
        fun: a <d> dimensional (noisy) function to minimize.
        x0: a matrix of <d> dimensional vectors of initial values, one per parallel problem.
        parallel_args: an optional sequence of arrays (of size equal to the number of parallel problems) of arguments
            to pass to the vectorized function to minimize. Parallel arguments are passed before other arguments.
        args: an optional sequence of arguments to pass to the function to minimize.
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        moves:
            - a single josiann.Move object
            - a sequence of josiann.Move objects (all Moves have the same probability of being selected at
                each step for proposing a new candidate vector x)
            - a sequence of tuples with the following format : (selection probability, josiann.Move)
                In this case, the selection probability dictates the probability of each Move of being selected at
                each step.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        alpha: cooling coefficient.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        backup: use Backup for storing previously computed function evaluations and reusing them when returning to
            the same position vector ? (Only available when using SetStep moves).
        seed: a seed for the random generator.
        verbose: print progress bar ?
        leave_progress_bar: leave progress bar after the algorithm has completed (only works if verbose) ?
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ?
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.
        dtype: the data type for the values stored in the Trace.

    Returns:
        A Result object.

    Notes:
        This algorithm is equivalent to running several independent :func:`josiann.sa` instances in parallel where the
        cost function would be the same but with different parameters.
        As for the vectorized version (:func:`josiann.vsa`), the cost function should accept a (n x d) matrix of
        position vectors in d-dimensions.
        Here, all `n` optimization problems are solved at once while still being treated as independent.

    """
    if seed is None:
        seed = int(time.time())

    params = initialize_sa(
        parallel_args,
        args,
        x0,
        max_iter,
        max_measures,
        alpha,
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
        progress_bar = trange(params.base.max_iter, unit="iteration", leave=leave_progress_bar)
    else:
        progress_bar = range(params.base.max_iter)

    # run the SA algorithm
    iteration = 0

    for iteration in progress_bar:
        temperature = T(iteration, params.base.T_0, params.base.alpha)
        current_n = n(iteration, params.base)
        current_sigma = sigma(iteration, params.base.T_0, params.base.alpha, params.base.epsilon)

        accepted = np.zeros(params.multi.nb_walkers, dtype=bool)
        explored = np.zeros((params.multi.nb_walkers, params.base.nb_dimensions))
        explored_costs = np.zeros(params.multi.nb_walkers)

        start = time.perf_counter()

        try:
            updates = vectorized_execution(
                params.fun,  # type: ignore[arg-type]
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

            for _walker_index, (_x, _cost, _state) in enumerate(updates):
                if _state is UpdateState.accepted:
                    # update positions, costs, n
                    x[_walker_index] = _x
                    costs[_walker_index] = _cost
                    last_ns[_walker_index] = current_n

                if _state in (UpdateState.accepted, UpdateState.rejected):
                    accepted[_walker_index] = _state is UpdateState.accepted
                    explored[_walker_index] = _x
                    explored_costs[_walker_index] = _cost

        except Exception:
            message = f"Unexpected failure while evaluating cost function : \n{traceback.format_exc()}"
            success = False
            break

        elapsed = time.perf_counter() - start

        trace.positions.store(iteration, x, np.array(costs), current_n, accepted, explored, explored_costs)
        trace.parameters.store(iteration, temperature, current_n, current_sigma, elapsed)

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
