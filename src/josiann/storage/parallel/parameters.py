from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
from attrs import frozen

import josiann.typing as jot
from josiann.backup.parallel.backup import ParallelBackup
from josiann.parallel.algorithms.compute import get_vectorized_mean_cost
from josiann.parallel.moves.base import ParallelMove, parse_moves
from josiann.storage.parameters import (
    BaseParameters,
    MoveParameters,
    MultiParameters,
    SAParameters,
    check_base_parameters_core,
    check_bounds,
)


@frozen(kw_only=True)
class ParallelBaseParameters(BaseParameters):
    """
    Object for storing the general parameters used for running the SA algorithm.
    """

    parallel_args: tuple[npt.NDArray[Any], ...]

    @property
    def nb_parallel_problems(self) -> int:
        return int(self.x0.shape[0])


@frozen(kw_only=True)
class ParallelSAParameters(SAParameters):
    """
    Object for storing the parameters used for running the SA algorithm.
    """

    base: ParallelBaseParameters


def check_base_parameters(
    parallel_args: Sequence[npt.NDArray[Any]] | None,
    args: tuple[Any, ...] | None,
    x0: npt.NDArray[Any],
    max_iter: int,
    max_measures: int,
    alpha: float,
    epsilon: float,
    T_0: float,
    tol: float,
    suppress_warnings: bool,
    detect_convergence: bool,
    dtype: jot.DType,
) -> tuple[ParallelBaseParameters, MultiParameters]:
    """
    Check validity of base parameters.

    Args:
        parallel_args: an optional sequence of arrays (of size equal to the number of parallel problems) of arguments
            to pass to the vectorized function to minimize. Parallel arguments are passed before other arguments.
        args: an optional sequence of arguments to pass to the function to minimize.
        x0: a <d> dimensional vector of initial values.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        alpha: cooling coefficient,
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        dtype: the data type for the values stored in the Trace.

    Returns:
        BaseParameters.
    """
    # arguments
    args = tuple(args) if args is not None else ()

    nb_parallel_problems = None

    if parallel_args is not None:
        for array in parallel_args:
            if not isinstance(array, np.ndarray):
                raise TypeError("All arrays in the parallel arguments must be numpy arrays.")

            if nb_parallel_problems is None:
                nb_parallel_problems = len(array)

            elif nb_parallel_problems != len(array):
                raise ValueError("Arrays of different lengths were found in the parallel arguments.")

    # initial values
    if x0.ndim == 1:
        if nb_parallel_problems is None:
            warn("Only one optimization problem has been defined, consider running the regular josiann.sa algorithm.")
            nb_parallel_problems = 1

        x0 = np.array([x0])

    elif nb_parallel_problems is None:
        nb_parallel_problems = len(x0)

    elif len(x0) != nb_parallel_problems:
        raise ValueError(
            f"'x0' defines {len(x0)} parallel problems while the parallel arguments define "
            f"{nb_parallel_problems} parallel problems."
        )

    T_0, alpha, max_iter, max_measures, sigma_max, x0 = check_base_parameters_core(
        T_0,
        epsilon,
        alpha,
        max_iter,
        max_measures,
        suppress_warnings,
        tol,
        x0,
        dtype,
    )

    return ParallelBaseParameters(
        parallel_args=() if parallel_args is None else tuple(arg[:, np.newaxis] for arg in parallel_args),
        args=args,
        x0=x0,
        max_iter=max_iter,
        max_measures=max_measures,
        epsilon=epsilon,
        T_0=T_0,
        tol=tol,
        alpha=alpha,
        sigma_max=sigma_max,
        suppress_warnings=suppress_warnings,
        detect_convergence=detect_convergence,
    ), MultiParameters(nb_walkers=nb_parallel_problems)


def initialize_sa(
    parallel_args: Sequence[npt.NDArray[Any]] | None,
    args: tuple[Any, ...] | None,
    x0: npt.NDArray[Any],
    max_iter: int,
    max_measures: int,
    alpha: float,
    epsilon: float,
    T_0: float,
    tol: float,
    moves: ParallelMove | Sequence[ParallelMove] | Sequence[tuple[float, ParallelMove]],
    bounds: tuple[float, float] | Sequence[tuple[float, float]] | None,
    fun: jot.PARALLEL_FUN_TYPE[...],
    backup: bool,
    suppress_warnings: bool,
    detect_convergence: bool,
    window_size: int | None,
    seed: int,
    dtype: jot.DType,
) -> ParallelSAParameters:
    """
    Check validity of parameters and compute initial values before running the SA algorithm.

    Args:
        parallel_args: an optional sequence of arrays (of size equal to the number of parallel problems) of arguments
            to pass to the vectorized function to minimize. Parallel arguments are passed before other arguments.
        args: an optional sequence of arguments to pass to the function to minimize.
        x0: a <d> dimensional vector of initial values or a matrix of initial values of shape (nb_walkers, d).
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        alpha: cooling coefficient.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        moves: either
                    - a single josiann.Move object
                    - a sequence of josiann.Move objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.Move)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        fun: a <d> dimensional vectorized (noisy) function to minimize.
        backup: use Backup for storing previously computed function evaluations and reusing them when returning to
            the same position vector ? (Only available when using SetStep moves).
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.
        seed: a seed for the random generator.
        dtype: the data type for the values stored in the Trace.

    Returns:
        Valid parameters and initial values.
    """
    np.random.seed(seed)

    # base parameters
    base_parameters, multi_parameters = check_base_parameters(
        parallel_args,
        args,
        x0,
        max_iter,
        max_measures,
        alpha,
        epsilon,
        T_0,
        tol,
        suppress_warnings,
        detect_convergence,
        dtype,
    )

    # bounds
    check_bounds(bounds, base_parameters.x0)

    # move parameters
    move_parameters = MoveParameters(*parse_moves(moves))
    move_parameters.set_bounds(bounds)

    # init backup storage
    backup_storage = ParallelBackup(
        active=move_parameters.using_SetMoves and backup,
        nb_parallel_problems=base_parameters.nb_parallel_problems,
    )

    # initial costs and last_ns
    costs = get_vectorized_mean_cost(
        fun,
        base_parameters.x0,
        1,
        base_parameters.parallel_args,
        base_parameters.args,
        [(0, 0.0) for _ in range(base_parameters.nb_parallel_problems)],
    )

    last_ns = [1 for _ in range(base_parameters.nb_parallel_problems)]

    # window size
    if window_size is not None:
        if max_iter < window_size < 1:
            raise ValueError(f"Invalid window size '{window_size}', should be in [{1}, {max_iter}].")

    else:
        window_size = max(50, int(0.1 * max_iter))

    return ParallelSAParameters(
        base=base_parameters,
        multi=multi_parameters,
        moves=move_parameters,
        fun=fun,
        backup=backup_storage,
        costs=costs,
        last_ns=last_ns,
        window_size=window_size,
        seed=seed,
    )
