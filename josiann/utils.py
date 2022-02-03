# coding: utf-8
# Created on 26/07/2021 12:08
# Author : matteo

"""
Classes:
    Trace: object for storing the SA history per walker.
    State: object for storing the state of the SA at a particular instant.
    Result: object for storing the final SA result.

Also defines general utils functions.
"""

# ====================================================
# imports
import numpy as np
from dataclasses import dataclass

from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .moves import Move


# ====================================================
# code
@dataclass
class State:
    """
    Object for describing the current state of the SA algorithm.

    complementary_set: set of complementary vectors x_[k] of shape (nb_walkers-1, ndim)
    iteration: current iteration number.
    max_iter: maximum iteration number.
    """
    complementary_set: np.ndarray
    iteration: int
    max_iter: int


# cost computation --------------------------------------------------------------------------------
def get_mean_cost(fun: Callable,
                  x: np.ndarray,
                  _n: int,
                  args: tuple,
                  previous_evaluations: tuple[int, float]) -> float:
    """
    Get the mean of <n> function evaluations for vector of values <x>.

    :param fun: a function to evaluate.
    :param x: a vector of values.
    :param _n: the number of evaluations to compute.
    :param args: arguments to be passed to <fun>.
    :param previous_evaluations: previously computed function evaluations at position x: number of last function
        evaluations and obtained mean.

    :return: the mean of function evaluations at x.
    """
    last_n, last_mean = previous_evaluations
    return last_mean * last_n / _n + sum([max(0, fun(x, *args)) for _ in range(_n - last_n)]) / _n


def get_evaluation_vectorized_mean_cost(fun: Callable,
                                        x: np.ndarray,
                                        _n: int,
                                        args: tuple,
                                        previous_evaluations: list[tuple[int, float]]) -> list[float]:
    """
    Same as 'get_mean_cost' but <fun> is a vectorized function and costs are computed for all walkers at once.

    :param fun: a vectorized function to evaluate.
    :param x: a matrix of position vectors of shape (nb_walkers, d).
    :param _n: the number of evaluations to compute.
    :param args: arguments to be passed to <fun>.
    :param previous_evaluations: list of previously computed function evaluations at position x: number of last function
        evaluations and obtained means for each walker position.

    :return: the mean of function evaluations at x.
    """
    evaluations = [0. for _ in range(len(x))]

    for walker_index, walker_position in enumerate(x):
        last_n, last_mean = previous_evaluations[walker_index]
        remaining_n = _n - last_n
        if remaining_n:
            evaluations[walker_index] = last_mean * last_n / _n + \
                sum(fun(np.tile(walker_position, (remaining_n, 1)), *args)) / _n

        else:
            evaluations[walker_index] = last_mean

    return evaluations


def get_walker_vectorized_mean_cost(fun: Callable,
                                    x: np.ndarray,
                                    _n: int,
                                    args: tuple,
                                    previous_evaluations: list[tuple[int, float]],
                                    vectorized_skip_marker: Any) -> list[float]:
    """
    Same as 'get_mean_cost' but <fun> is a vectorized function and costs are computed for all walkers at once but
        sequentially on function evaluations.

    :param fun: a vectorized function to evaluate.
    :param x: a matrix of position vectors of shape (nb_walkers, d).
    :param _n: the number of evaluations to compute.
    :param args: arguments to be passed to <fun>.
    :param previous_evaluations: list of previously computed function evaluations at position x: number of last function
        evaluations and obtained means for each walker position.
    :param vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
        evaluation for a particular position vector can be skipped.

    :return: the mean of function evaluations at x.
    """
    zipped_last = zip(*[previous_evaluations[walker_index] for walker_index, _ in enumerate(x)])
    last_n = list(next(zipped_last))
    remaining_n = [_n - ln for ln in last_n]
    last_mean = list(next(zipped_last))

    if max(remaining_n):
        costs = np.zeros(len(x))

        for eval_index in range(max(remaining_n)):
            eval_vector = np.array([walker_position if eval_index < remaining_n[walker_index] else
                                    vectorized_skip_marker
                                    for walker_index, walker_position in enumerate(x)])

            res = np.array(fun(eval_vector, *args))

            for walker_index, _ in enumerate(res):
                if eval_index >= remaining_n[walker_index]:
                    res[walker_index] = 0.

            costs += res

        return (np.array(last_mean) * last_n + costs) / _n

    return last_mean


# parameters computation --------------------------------------------------------------------------
def acceptance_log_probability(current_cost: float,
                               new_cost: float,
                               _T: float) -> float:
    """
    Compute the acceptance probability for a new proposed cost, given the current cost and a temperature.

    :param current_cost: the current cost.
    :param new_cost: the new proposed cost.
    :param _T: the current temperature.

    :return: the probability of acceptance of the new proposed cost.
    """
    return (current_cost - new_cost) / _T


def sigma(k: int,
          T_0: float,
          alpha: float,
          epsilon: float) -> float:
    """
    Compute the estimated standard deviation at iteration k.

    :param k: the iteration number.
    :param T_0: initial temperature value.
    :param alpha: rate of temperature decrease.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)

    :return: the estimated standard deviation.
    """
    return T_0 * (alpha * (1 - epsilon)) ** k


def n(k: int,
      max_measures: int,
      sigma_max: float,
      T_0: float,
      alpha: float,
      epsilon: float) -> int:
    """
    Compute the number of necessary measures at iteration k.

    :param k: the iteration number.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param sigma_max: the maximal values (reached at iteration 0) for sigma.
    :param T_0: initial temperature value.
    :param alpha: rate of temperature decrease.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)

    :return: the number of necessary measures.
    """
    if max_measures == 1:
        return 1

    return int(np.ceil((max_measures * sigma_max ** 2) /
                       ((max_measures - 1) * sigma(k, T_0, alpha, epsilon) ** 2 + sigma_max ** 2)))


def T(k: int,
      T_0: float,
      alpha: float) -> float:
    """
    Compute the temperature at iteration k.

    :param k: the iteration number.
    :param T_0: initial temperature value.
    :param alpha: rate of temperature decrease.

    :return: the temperature.
    """
    return T_0 * alpha ** k


def get_slots_per_walker(slots: int,
                         nb_walkers: int) -> list[int]:
    """
    Assign to each walker an approximately equal number of slots.

    :param slots: the total number of available slots.
    :param nb_walkers: the number of walkers.

    :return: the number of slots per walker.
    """
    per_walker, plus_one = divmod(slots, nb_walkers)

    return [per_walker + 1 for _ in range(plus_one)] + [per_walker for _ in range(nb_walkers - plus_one)]


def get_exploration_plan(acceptance: float,
                         nb_slots: int,
                         x: np.ndarray,
                         list_moves: list['Move'],
                         list_probabilities: list[float],
                         state: State,
                         plan: list[np.ndarray]) -> tuple[int, int, list[np.ndarray]]:
    """
    Define an exploration plan : a set of moves (of size <nb_slots>) to take from an initial position vector <x>.
    For low acceptance values, the moves are mostly picked around the original position vector in a star pattern.
    For high acceptance values, since the probability of a move being accepted is higher, moves are taken more
        sequentially.

    :param acceptance: the current proportion of accepted moves.
    :param nb_slots: the size of the exploration plan (number of moves to take).
    :param x: the current position vector of shape (ndim,)
    :param list_moves: the list of possible moves from which to choose.
    :param list_probabilities: the associated probabilities.
    :param state: the current state of the SA algorithm.
    :param plan: the exploration plan, as a list of position vectors.

    :return: the number of used slots, the number of slots left, the exploration plan.
    """
    used = max(int(np.ceil((1-acceptance) * nb_slots)), 1)
    left = nb_slots - used

    for _ in range(used):
        move = np.random.choice(list_moves, p=list_probabilities)
        proposal = move.get_proposal(x, state)
        plan.append(proposal)

        if left > 0:
            used_after, *_ = get_exploration_plan(acceptance, int(np.ceil(left/used)), proposal,
                                                  list_moves, list_probabilities,
                                                  State(state.complementary_set, state.iteration + 1, state.max_iter),
                                                  plan)
            left -= used_after

    return used, left, plan
