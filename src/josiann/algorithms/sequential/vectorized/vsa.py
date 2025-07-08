from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm.autonotebook import tqdm, trange

import josiann.typing as jot
from josiann.algorithms.map.vectorized import vectorized_execution
from josiann.algorithms.run import run_simulated_annealing
from josiann.algorithms.sequential.vectorized.intialize import initialize_vsa
from josiann.moves.base import Move
from josiann.moves.sequential import RandomStep
from josiann.storage.result import Result
from josiann.storage.trace import OneTrace


def vsa(
    fun: Callable[..., list[float]],
    x0: npt.NDArray[Any],
    args: tuple[Any, ...] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    moves: Move | Sequence[Move] | Sequence[tuple[float, Move]] = (
        (0.8, RandomStep(magnitude=0.05)),
        (0.2, RandomStep(magnitude=0.5)),
    ),
    nb_walkers: int = 1,
    max_iter: int = 200,
    max_measures: int = 20,
    alpha: float = 0.95,
    epsilon: float = 0.01,
    T_0: float = 5.0,
    tol: float = 1e-3,
    vectorized_on_evaluations: bool = True,
    vectorized_skip_marker: Any = None,
    backup: bool = False,
    nb_slots: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
    suppress_warnings: bool = False,
    detect_convergence: bool = True,
    window_size: int | None = None,
    dtype: jot.DType = np.float64,  # type: ignore[assignment]
) -> Result:
    """
    Simulated Annealing using vectorized cost functions for computing multiple function evaluations at once.

    Args:
        fun: a <d> dimensional (noisy) function to minimize.
        x0: a <d> dimensional vector of initial values.
        args: an optional sequence of arguments to pass to the function to minimize.
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        moves:
            - a single josiann.Move object
            - a sequence of josiann.Move objects (all Moves have the same probability of being selected at each step
                for proposing a new candidate vector x)
            - a sequence of tuples with the following format : (selection probability, josiann.Move)
                In this case, the selection probability dictates the probability of each Move of being selected at
                each step.
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        alpha: cooling coefficient.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        vectorized_on_evaluations: the vectorization can happen on walkers or on function evaluations.

             - On function evaluations, a loop on walkers calls <fun> with a vector of positions of the walker, repeated
                for the number of needed function evaluations.
                Ex: 2 walkers with position vectors p1 and p2 each need n1 and n2 function evaluations. <fun> is first
                called for walker 1 with a vector (p1, p1, ..., p1) of size n1, then <fun> is called for walker 2 with a
                vector (p2, p2, ..., p2) of size n2.
                This is the default option and is valid when <fun> is ok with receiving vectors of varying length and
                when <max_measures> is greater than <nb_walkers>.

             - On walkers, a loop on function evaluations calls <fun> with a vector of fixed size = <nb_walkers>.
                Ex: 2 walkers with position vectors p1 and p2 each need n1 and n2 function evaluations. <fun> is called
                with vector (p1, p2) for max(n1, n2) times.
                Often, n1 =/= n2 which would yield unnecessary function evaluations (e.g. when n1 < n2, some evaluations
                of p1 are not needed while p2 is still evaluated). To indicated that to <fun>, the
                <vectorized_skip_marker> is passed instead of unnecessary position vectors (e.g. when n1 < n2,
                the vector passed to <fun> will eventually be (<vectorized_skip_marker>, p2) instead of (p1, p2)).
                This is valid when <fun> needs to receive vectors of fixed length and when <nb_walkers> is greater than
                <max_measures>.
        vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
            evaluation for a particular position vector can be skipped.
        backup: use Backup for storing previously computed function evaluations and reusing them when returning to
            the same position vector ? (Only available when using SetStep moves).
        nb_slots: When using a vectorized function, the total number of position vectors for which the cost can be
            computed at once.
            For example, when using 5 walkers and 22 slots, each walker will be attributed respectively 5, 5, 4, 4,
            and 4 slots.
            Multiple slots per walker are used for exploring multiple possible moves from the starting position vector
            of each walker, increasing convergence speed.
        seed: a seed for the random generator.
        verbose: print progress bar ?
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ?
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.
        dtype: the data type for the values stored in the Trace.

    Returns:
        A Result object.
    """
    if seed is None:
        seed = int(time.time())

    params = initialize_vsa(
        args,
        x0,
        nb_walkers,
        max_iter,
        max_measures,
        alpha,
        epsilon,
        T_0,
        tol,
        moves,
        bounds,
        fun,
        vectorized_on_evaluations,
        vectorized_skip_marker,
        backup,
        nb_slots,
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
        vectorized_execution,
        np.array(last_ns),
        nb_walkers,
        params,
        progress_bar,
        trace,
        x,
        nb_slots=params.multi.nb_slots_per_walker,
        vectorized_on_evaluations=params.multi.vectorized_on_evaluations,
        vectorized_skip_marker=params.multi.vectorized_skip_marker,
    )
