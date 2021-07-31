# coding: utf-8
# Created on 23/07/2021 23:28
# Author : matteo

# ====================================================
# imports
import numpy as np
from tqdm import tqdm

from typing import Callable, Tuple, Optional, Sequence, Union, Any, List

from .utils import Result, Trace, get_mean_cost, acceptance_log_probability, check_parameters, parse_moves, n, T, sigma
from .moves import Move, RandomStep


# ====================================================
# code
def initialize_sa(args: Optional[Sequence],
                  x0: np.ndarray,
                  nb_walkers: int,
                  max_iter: int,
                  max_measures: int,
                  final_acceptance_probability: float,
                  epsilon: float,
                  T_0: Optional[float],
                  tol: float,
                  window_size: int,
                  moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]],
                  bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]],
                  fun: Callable[[np.ndarray, Any], float]) -> Tuple[
    Tuple, np.ndarray, int, int, float, float, float, float, int, bool, List[float], List[Move], np.ndarray,
    List[float], List[int], float, float, float
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
    :param T_0: optional initial temperature value.
    :param tol: the convergence tolerance.
    :param window_size: a window of the last <window_size> cost values are used to test for convergence.
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

    :return: Valid parameters and initial values.
    """
    # check parameters
    args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        computed_T_0 = check_parameters(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability,
                                        epsilon, T_0, tol, window_size, bounds)

    # get moves and associated probabilities
    list_probabilities, list_moves = parse_moves(moves, nb_walkers)

    for move in list_moves:
        move.set_bounds(bounds)

    # initial state
    x = x0.astype(np.float32)
    costs = [get_mean_cost(fun, x_vector, 1, *args) for x_vector in x]
    last_ns = [1 for _ in range(nb_walkers)]

    # initialize parameters
    T_final = -1 / np.log(final_acceptance_probability) / 10
    alpha = (T_final / T_0) ** (1 / max_iter)
    sigma_max = T_0

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        computed_T_0, list_probabilities, list_moves, x, costs, last_ns, T_final, alpha, sigma_max


def sa(fun: Callable[[np.ndarray, Any], float],
       x0: np.ndarray,
       args: Optional[Sequence] = None,
       bounds: Optional[Sequence[Tuple[float, float]]] = None,
       moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]] = ((0.8, RandomStep(0.05)),
                                                                           (0.2, RandomStep(0.5))),
       nb_walkers: int = 1,
       max_iter: int = 200,
       max_measures: int = 20,
       final_acceptance_probability: float = 1e-5,
       epsilon: float = 0.01,
       T_0: Optional[float] = None,
       tol: float = 1e-3,
       window_size: int = 100) -> Result:
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
    :param T_0: optional initial temperature value.
    :param tol: the convergence tolerance.
    :param window_size: a window of the last <window_size> cost values are used to test for convergence.

    :return: a Result object.
    """
    args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, computed_T_0, \
        list_probabilities, list_moves, x, costs, last_ns, T_final, alpha, sigma_max = \
        initialize_sa(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol,
                      window_size, moves, bounds, fun)

    # initialize the trace history keeper
    trace = Trace(max_iter, x0.shape, window_size=window_size)
    trace.initialize(x, costs)

    progress_bar = tqdm(range(max_iter), unit='iteration')

    # run the SA algorithm
    for iteration in progress_bar:

        temperature = T(iteration, T_0, alpha)
        current_n = n(iteration, max_measures, sigma_max, T_0, alpha, epsilon)
        accepted = [False for _ in range(nb_walkers)]

        for walker_index in range(nb_walkers):

            # pick a move at random from available moves
            move = np.random.choice(list_moves, p=list_probabilities)

            # generate a new proposal as a neighbor of x and get its cost
            proposed_x = move.get_proposal(x[walker_index], c=np.delete(x, walker_index),
                                           iteration=iteration, max_iter=max_iter)
            proposed_cost = get_mean_cost(fun, proposed_x, current_n, *args)

            if acceptance_log_probability(costs[walker_index] * last_ns[walker_index] / current_n, proposed_cost,
                                          temperature) > np.log(np.random.random()):
                x[walker_index], costs[walker_index] = proposed_x, proposed_cost
                last_ns[walker_index] = current_n
                accepted[walker_index] = True

        trace.store(x, costs, temperature, current_n, sigma(iteration, T_0, alpha, epsilon), accepted)

        progress_bar.set_description(f"T: {temperature:.4f}    %A: {trace.acceptance_fraction()[0]*100:.4f}%    Best: "
                                     f"{trace.get_best()[1]:.4f} Current: {np.min(costs):.4f}")

        if trace.reached_convergence(tol):
            message = 'Convergence tolerance reached.'
            break

    else:
        message = 'Requested number of iterations reached.'

    trace.finalize()

    return Result(message, True, trace, args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0,
                  tol, window_size, alpha, T_final, computed_T_0)
