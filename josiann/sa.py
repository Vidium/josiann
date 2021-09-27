# coding: utf-8
# Created on 23/07/2021 23:28
# Author : matteo

"""
Core Simulated Annealing function.
"""

# ====================================================
# imports
import time
import traceback
import numpy as np
from tqdm import tqdm
from warnings import warn
from itertools import repeat

from typing import Callable, Tuple, Optional, Sequence, Union, Any, List, Type

from .utils import Result, Trace, get_mean_cost, get_vectorized_mean_cost, check_parameters, get_slots_per_walker, n, \
    T, sigma
from .moves import Move, RandomStep, SetStep, SetStretch, parse_moves
from .__mappers import LinearExecutor, VectorizedExecutor, ParallelExecutor
from .__backup import Backup


# ====================================================
# code
def __initialize_sa(args: Optional[Sequence],
                    x0: np.ndarray,
                    nb_walkers: int,
                    max_iter: int,
                    max_measures: int,
                    final_acceptance_probability: float,
                    epsilon: float,
                    T_0: float,
                    tol: float,
                    moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]],
                    bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]],
                    fun: Callable[[np.ndarray, Any], Union[List[float], float]],
                    nb_cores: int,
                    vectorized: bool,
                    backup: bool,
                    nb_slots: Optional[int],
                    suppress_warnings: bool) -> Tuple[
    Tuple, np.ndarray, int, int, float, float, float, float, int, List[float], List[Move], np.ndarray,
    List[float], List[int], float, float, float, int, List[int], Backup
]:
    """
    Check validity of parameters and compute initial values before running the SA algorithm.

    :param args: an optional sequence of arguments to pass to the function to minimize.
    :param x0: a <d> dimensional vector of initial values or a matrix of initial values of shape (nb_walkers, d).
    :param nb_walkers: the number of parallel walkers in the ensemble.
    :param max_iter: the maximum number of iterations before stopping the algorithm.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)
    :param T_0: initial temperature value.
    :param tol: the convergence tolerance.
    :param moves: either
                    - a single josiann.Move object
                    - a sequence of josiann.Move objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.Move)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
        or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
    :param fun: a <d> dimensional (noisy) function to minimize.
    :param nb_cores: number of cores that can be used to move walkers in parallel.
    :param vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
        just one. (<nb_cores> parameter will be set to 1 in this case.)
    :param backup: use Backup for storing previously computed function evaluations and reusing them when returning to
        the same position vector ? (Only available when using SetStep moves).
    :param nb_slots: When using a vectorized function, the total number of position vectors for which the cost can be
        computed at once.
    :param suppress_warnings: remove warnings ?

    :return: Valid parameters and initial values.
    """
    # check parameters
    args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, nb_cores \
        = check_parameters(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability, epsilon, T_0,
                           tol, bounds, nb_cores, vectorized)

    window_size = max(1, min(50, int(0.1 * max_iter)))

    if not suppress_warnings and max_iter < 200:
        warn('It is not recommended running the SA algorithm with less than 200 iterations.')

    # get moves and associated probabilities
    list_probabilities, list_moves = parse_moves(moves, nb_walkers)

    using_SetStep = False

    for move in list_moves:
        if isinstance(move, (SetStep, SetStretch)):
            using_SetStep = True

        move.set_bounds(bounds)

    # init backup storage
    backup_storage = Backup(active=using_SetStep and backup)

    # initial state
    x = x0.astype(np.float32)

    if vectorized:
        costs = get_vectorized_mean_cost(fun, x, 1, args, [(0, 0.) for _ in range(len(x))])
    else:
        costs = [get_mean_cost(fun, x_vector, 1, args, (0, 0.)) for x_vector in x]

    last_ns = [1 for _ in range(nb_walkers)]

    # initialize parameters
    T_final = -1 / np.log(final_acceptance_probability)
    alpha = (T_final / T_0) ** (1 / max_iter)

    # sigma_max = T_0
    sigma_max = np.sqrt((max_measures - 1) * T_0 * alpha * (1 - epsilon)) / 3

    if nb_slots is None:
        nb_slots_per_walker = [1 for _ in range(nb_walkers)]

    elif not vectorized:
        raise ValueError("Cannot use slots unless using a vectorized cost function.")

    elif nb_slots < nb_walkers:
        raise ValueError(f"nb_slots ({nb_slots}) is less than the number of walkers ({nb_walkers})!")

    else:
        nb_slots_per_walker = get_slots_per_walker(nb_slots, nb_walkers)

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        list_probabilities, list_moves, x, costs, last_ns, T_final, alpha, sigma_max, nb_cores, nb_slots_per_walker, \
        backup_storage


def sa(fun: Callable[[np.ndarray, Any], Union[float, List[float]]],
       x0: np.ndarray,
       args: Optional[Sequence] = None,
       bounds: Optional[Sequence[Tuple[float, float]]] = None,
       moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]] = ((0.8, RandomStep(0.05)),
                                                                           (0.2, RandomStep(0.5))),
       nb_walkers: int = 1,
       max_iter: int = 200,
       max_measures: int = 20,
       final_acceptance_probability: float = 1e-300,
       epsilon: float = 0.01,
       T_0: float = 5.,
       tol: float = 1e-3,
       nb_cores: int = 1,
       vectorized: bool = False,
       backup: bool = False,
       nb_slots: Optional[int] = None,
       seed: int = 42,
       verbose: bool = True,
       suppress_warnings: bool = False) -> Result:
    """
    Simulated Annealing for minimizing noisy cost functions.

    :param fun: a <d> dimensional (noisy) function to minimize.
    :param x0: a <d> dimensional vector of initial values.
    :param args: an optional sequence of arguments to pass to the function to minimize.
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
        or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
    :param moves: either
                    - a single josiann.SingleMove object
                    - a sequence of josiann.SingleMove objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.SingleMove)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
    :param nb_walkers: the number of parallel walkers in the ensemble.
    :param max_iter: the maximum number of iterations before stopping the algorithm.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)
    :param T_0: initial temperature value.
    :param tol: the convergence tolerance.
    :param nb_cores: number of cores that can be used to move walkers in parallel.
    :param vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
        just one. (<nb_cores> parameter will be set to 1 in this case.)
    :param backup: use Backup for storing previously computed function evaluations and reusing them when returning to
        the same position vector ? (Only available when using SetStep moves).
    :param nb_slots: When using a vectorized function, the total number of position vectors for which the cost can be
        computed at once.
        For example, when using 5 walkers and 22 slots, each walker will be attributed respectively 5, 5, 4, 4,
        and 4 slots.
        Multiple slots per walker are used for exploring multiple possible moves from the starting position vector of
        each walker, increasing convergence speed.
    :param seed: a seed for the random generator.
    :param verbose: print progress bar ? (default True)
    :param suppress_warnings: remove warnings ? (default False)

    :return: a Result object.
    """
    np.random.seed(seed)

    args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        list_probabilities, list_moves, x, costs, last_ns, T_final, alpha, sigma_max, nb_cores, nb_slots_per_walker, \
        backup_storage = \
        __initialize_sa(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol,
                        moves, bounds, fun, nb_cores, vectorized, backup, nb_slots, suppress_warnings)

    # initialize the trace history keeper
    trace = Trace(max_iter, x0.shape, window_size=window_size)
    trace.initialize(x, costs)

    if verbose:
        progress_bar = tqdm(range(max_iter), unit='iteration')
    else:
        progress_bar = range(max_iter)

    if vectorized:
        executor: Union[Type[VectorizedExecutor], Type[ParallelExecutor], Type[LinearExecutor]] = VectorizedExecutor
    elif nb_cores > 1:
        executor = ParallelExecutor
    else:
        executor = LinearExecutor

    # run the SA algorithm
    with executor(max_workers=nb_cores) as ex:
        try:
            for iteration in progress_bar:
                start = time.time()

                temperature = T(iteration, T_0, alpha)
                current_n = n(iteration, max_measures, sigma_max, T_0, alpha, epsilon)
                accepted = [False for _ in range(nb_walkers)]
                rescued = [False for _ in range(nb_walkers)]

                acceptance_fraction = trace.mean_acceptance_fraction()

                updates = ex.map(fun,
                                 x.copy(),
                                 costs,
                                 repeat(current_n),
                                 last_ns,
                                 repeat(args),
                                 repeat(list_moves),
                                 repeat(list_probabilities),
                                 repeat(iteration),
                                 repeat(max_iter),
                                 repeat(temperature),
                                 repeat(nb_slots_per_walker),
                                 repeat(acceptance_fraction if acceptance_fraction is not np.nan else 1.),
                                 positions=x.copy(),
                                 backup=backup_storage)

                for _x, _cost, _last_n, _accepted, _walker_index in updates:
                    if _accepted:
                        x[_walker_index] = _x
                        costs[_walker_index] = _cost
                        last_ns[_walker_index] = _last_n
                        accepted[_walker_index] = _accepted

                index = trace.store(x, costs, temperature, current_n, sigma(iteration, T_0, alpha, epsilon), accepted)

                best_position, best_cost, best_index = trace.get_best()
                stuck_walkers = trace.are_stuck()

                for _walker_index in range(nb_walkers):
                    # rescue stuck walkers
                    if stuck_walkers[_walker_index]:
                        x[_walker_index] = best_position
                        costs[_walker_index] = best_cost
                        last_ns[_walker_index] = n(best_index[0], max_measures, sigma_max, T_0, alpha, epsilon)
                        accepted[_walker_index] = True
                        rescued[_walker_index] = True

                trace.update(index, x, costs, rescued, best_position, best_cost, best_index, time.time() - start)

                if verbose:
                    progress_bar.set_description(f"T: "
                                                 f"{temperature:.4f}"
                                                 f"  A: "
                                                 f"{trace.mean_acceptance_fraction()*100:.4f}%"
                                                 f"  Best: "
                                                 f"{trace.get_best()[1]:.4f}"
                                                 f"  Current: "
                                                 f"{np.min(costs):.4f}")

                if trace.reached_convergence(tol):
                    message, success = 'Convergence tolerance reached.', True
                    break

            else:
                message, success = 'Requested number of iterations reached.', False

        # noinspection PyBroadException
        except Exception:
            message, success = f'Unexpected failure : \n{traceback.format_exc()}', False

    trace.finalize()

    return Result(message, success, trace, args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0,
                  sigma_max, tol, window_size, alpha, T_final, nb_cores, vectorized, backup_storage.active)
