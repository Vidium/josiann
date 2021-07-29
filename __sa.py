# coding: utf-8
# Created on 29/07/2021 09:45
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import Optional, Sequence, Union, Tuple, Callable, Any, List
from typing_extensions import Literal

from .name_utils import ShapeError
from .utils import parse_moves, get_mean_cost
from .__moves import Move


# ====================================================
# code
def get_delta_max() -> float:
    """TODO"""
    raise NotImplementedError


def check_parameters(args: Optional[Sequence],
                     x0: np.ndarray,
                     nb_walkers: int,
                     max_iter: int,
                     max_measures: int,
                     final_acceptance_probability: float,
                     epsilon: float,
                     T_0: Optional[float],
                     tol: float,
                     window_size: int,
                     mtype: Literal['SingleMove', 'EnsembleMove']) -> Tuple[Tuple, np.ndarray, int, int, float, float,
                                                                            float, float, int, bool]:
    """
    Check validity of parameters.

    :param args: an optional sequence of arguments to pass to the function to minimize.
    :param x0: a <d> dimensional vector of initial values.
    :param nb_walkers: the number of parallel walkers in the ensemble.
    :param max_iter: the maximum number of iterations before stopping the algorithm.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)
    :param T_0: optional initial temperature value.
    :param tol: the convergence tolerance.
    :param window_size: a window of the last <window_size> cost values are used to test for convergence.
    :param mtype: the type of moves that are accepted (either 'SingleMove' or 'EnsembleMove').

    :return: Valid parameters.
    """
    args = args if args is not None else ()

    if mtype == 'SingleMove':
        if x0.ndim != 1:
            raise ShapeError(f'Vector of initial values should be one dimensional, not {x0.ndim}.')

        x0 = np.array([x0])

    if x0.shape[0] != nb_walkers:
        raise ShapeError(f'Matrix of initial values should have {nb_walkers} dimensions (equal to the number of '
                         f'parallel walkers), not {x0.shape[1]}')

    if max_iter < 0:
        raise ValueError("'max_iter' parameter must be positive.")
    else:
        max_iter = int(max_iter)

    if max_measures < 0:
        raise ValueError("'max_measures' parameter must be positive.")
    else:
        max_measures = int(max_measures)

    if final_acceptance_probability < 0 or final_acceptance_probability > 1:
        raise ValueError(f"Invalid value '{final_acceptance_probability}' for 'final_acceptance_probability', "
                         f"should be in [0, 1].")

    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"Invalid value '{epsilon}' for 'epsilon', should be in (0, 1).")

    if T_0 is not None and T_0 < 0:
        raise ValueError("'T_0' parameter must be at least 0.")

    if T_0 is None:
        T_0 = -get_delta_max() / np.log(0.8)
        computed_T_0 = True
    else:
        T_0 = float(T_0)
        computed_T_0 = False

    if tol <= 0:
        raise ValueError("'tol' parameter must be strictly positive.")

    if window_size < 1:
        raise ValueError("'window_size' parameter must be greater than 0.")
    else:
        window_size = int(window_size)

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, computed_T_0


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
                  bounds: Optional[Sequence[Tuple[float, float]]],
                  fun: Callable[[np.ndarray, Any], float],
                  mtype: Literal['SingleMove', 'EnsembleMove']) -> Tuple[
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
    :param mtype: the type of moves that are accepted (either 'SingleMove' or 'EnsembleMove').

    :return: Valid parameters and initial values.
    """
    # check parameters
    args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        computed_T_0 = check_parameters(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability,
                                        epsilon, T_0, tol, window_size, mtype=mtype)

    # get moves and associated probabilities
    list_probabilities, list_moves = parse_moves(moves, mtype=mtype)

    for move in list_moves:
        move.set_bounds(bounds)

    # initial state
    x = x0.copy()
    costs = [get_mean_cost(fun, x_vector, 1, *args) for x_vector in x]
    last_ns = [1 for _ in range(nb_walkers)]

    # initialize parameters
    T_final = -1 / np.log(final_acceptance_probability)
    alpha = (T_final / T_0) ** (1 / max_iter)
    sigma_max = T_0

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        computed_T_0, list_probabilities, list_moves, x, costs, last_ns, T_final, alpha, sigma_max
