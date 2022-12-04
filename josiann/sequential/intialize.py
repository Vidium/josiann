# coding: utf-8
# Created on 03/12/2022 17:25
# Author : matteo

# ====================================================
# imports
from multiprocessing import cpu_count

import numpy as np
from attrs import frozen

from typing import Sequence

from josiann.backup import SequentialBackup
from josiann.compute import get_mean_cost
from josiann.moves import Move
from josiann.moves import parse_moves
from josiann.storage.parameters import MoveParameters
from josiann.storage.parameters import ParallelParameters
from josiann.storage.parameters import SAParameters
from josiann.storage.parameters import check_base_parameters
from josiann.storage.parameters import check_bounds
from josiann.typing import FUN_TYPE


# ====================================================
# code
@frozen(kw_only=True)
class SequentialParallelParameters(ParallelParameters):
    """
    Object for storing the parameters managing the calls to parallel or vectorized cost functions.
    """
    nb_cores: int


@frozen(kw_only=True)
class SequentialSAParameters(SAParameters):
    """
    Object for storing the parameters used for running the SA algorithm.
    """
    parallel: SequentialParallelParameters


def check_parallel_parameters(nb_walkers: int,
                              nb_cores: int) -> SequentialParallelParameters:
    """
    Check validity of parallel parameters.

    Args:
        nb_walkers: the number of parallel walkers in the ensemble.
        nb_cores: number of cores that can be used to move walkers in parallel.

    Returns:
        ParallelParameters.
    """
    # nb_cores
    nb_cores = int(nb_cores)

    if nb_cores < 1:
        raise ValueError('Cannot use less than one core.')

    if nb_cores > cpu_count():
        raise ValueError(f"Cannot use more than available CPUs ({cpu_count()}).")

    return SequentialParallelParameters(nb_walkers=nb_walkers,
                                        nb_cores=nb_cores)


def initialize_sa(args: Sequence | None,
                  x0: np.ndarray,
                  nb_walkers: int,
                  max_iter: int,
                  max_measures: int,
                  final_acceptance_probability: float,
                  epsilon: float,
                  T_0: float,
                  tol: float,
                  moves: Move | Sequence[Move] | Sequence[tuple[float, Move]],
                  bounds: tuple[float, float] | Sequence[tuple[float, float]] | None,
                  fun: FUN_TYPE,
                  nb_cores: int,
                  backup: bool,
                  suppress_warnings: bool,
                  detect_convergence: bool,
                  window_size: int | None,
                  seed: int,
                  dtype: type[np.dtype] | type[np.number]) -> SequentialSAParameters:
    """
    Check validity of parameters and compute initial values before running the SA algorithm.

    Args:
        args: an optional sequence of arguments to pass to the function to minimize.
        x0: a <d> dimensional vector of initial values or a matrix of initial values of shape (nb_walkers, d).
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
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
        fun: a <d> dimensional (noisy) function to minimize.
        nb_cores: number of cores that can be used to move walkers in parallel.
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
    base_parameters = check_base_parameters(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability,
                                            epsilon, T_0, tol, suppress_warnings, detect_convergence, dtype)

    # parallel parameters
    parallel_parameters = check_parallel_parameters(nb_walkers, nb_cores)

    # bounds
    check_bounds(bounds, base_parameters.x0)

    # move parameters
    move_parameters = MoveParameters(*parse_moves(moves, nb_walkers))
    move_parameters.set_bounds(bounds)

    # init backup storage
    backup_storage = SequentialBackup(active=move_parameters.using_SetMoves and backup)

    # initial costs and last_ns
    costs = [get_mean_cost(fun, x_vector, 1, base_parameters.args, (0, 0.)) for x_vector in base_parameters.x0]

    last_ns = [1 for _ in range(nb_walkers)]

    # window size
    if window_size is not None:
        if max_iter < window_size < 1:
            raise ValueError(f"Invalid window size '{window_size}', should be in [{1}, {max_iter}].")

    else:
        window_size = max(50, int(0.1 * max_iter))

    return SequentialSAParameters(base=base_parameters,
                                  parallel=parallel_parameters,
                                  moves=move_parameters,
                                  fun=fun,
                                  backup=backup_storage,
                                  costs=costs,
                                  last_ns=last_ns,
                                  window_size=window_size,
                                  seed=seed)
