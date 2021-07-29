# coding: utf-8
# Created on 29/07/2021 08:57
# Author : matteo

# ====================================================
# imports
import numpy as np
from tqdm import tqdm

from typing import Callable, Any, Optional, Sequence, Tuple, Union

from ..utils import Result, Trace, n, T, sigma, get_mean_cost, acceptance_probability
from ..__sa import initialize_sa
from .moves import EnsembleMove, Stretch


# ====================================================
# code
def ensemble_sa(fun: Callable[[np.ndarray, Any], float],
                x0: np.ndarray,
                args: Optional[Sequence] = None,
                bounds: Optional[Sequence[Tuple[float, float]]] = None,
                moves: Union[EnsembleMove, Sequence[EnsembleMove], Sequence[Tuple[float, EnsembleMove]]] = Stretch(),
                nb_walkers: int = 10,
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
    :param x0: a matrix of initial values of shape (nb_walkers, d).
    :param args: an optional sequence of arguments to pass to the function to minimize.
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
        or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
    :param moves: either
                    - a single josiann.EnsembleMove object
                    - a sequence of josiann.EnsembleMove objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.EnsembleMove)
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
        list_probabilities, list_moves, x, cost, last_n, T_final, alpha, sigma_max = \
        initialize_sa(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol,
                      window_size, moves, bounds, fun, mtype='EnsembleMove')

    # initialize the trace history keeper
    trace = Trace(max_iter, len(x0), window_size=window_size)
    trace.initialize(x, cost)

    progress_bar = tqdm(range(max_iter), unit='iteration')

    # run the SA algorithm
    for iteration in progress_bar:

        temperature = T(iteration, T_0, alpha)
        current_n = n(iteration, max_measures, sigma_max, T_0, alpha, epsilon)
        accepted = False

        # pick a move at random from available moves
        move = np.random.choice(list_moves, p=list_probabilities)

        # generate a new proposal as a neighbor of x and get its cost
        proposed_x = move.get_proposal(x)
        proposed_cost = get_mean_cost(fun, proposed_x, current_n, *args)

        if acceptance_probability(cost * last_n / current_n, proposed_cost, temperature) > np.log(np.random.random()):
            x, cost = proposed_x, proposed_cost
            last_n = current_n
            accepted = True

        trace.store(x, cost, temperature, current_n, sigma(iteration, T_0, alpha, epsilon), accepted)

        progress_bar.set_description(f"T: {temperature:.4f}    %A: {trace.acceptance_fraction() * 100:.4f}%    Best: "
                                     f"{trace.get_best()[1]:.4f} Current: {cost:.4f}")

        if trace.reached_convergence(tol):
            message = 'Convergence tolerance reached.'
            break

    else:
        message = 'Requested number of iterations reached.'

    trace.finalize()

    return Result(message, True, trace, args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0,
                  tol, window_size, alpha, T_final, computed_T_0)
