# coding: utf-8
# Created on 03/08/2021 12:21
# Author : matteo

# ====================================================
# imports
import numpy as np
from itertools import repeat
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from typing import Callable, Any, Tuple, List, Iterable, Sequence, Iterator, cast

from .utils import State, get_mean_cost, get_vectorized_mean_cost, acceptance_log_probability
from .moves import Move
from .__backup import Backup, BackupManager


# ====================================================
# code
def _update_walker(fun: Callable[[np.ndarray, Any], float],
                   x: np.ndarray,
                   cost: float,
                   current_n: int,
                   last_n: int,
                   args: Tuple,
                   list_moves: List[Move],
                   list_probabilities: List[float],
                   iteration: int,
                   max_iter: int,
                   temperature: float,
                   complementary_set: np.ndarray,
                   backup_storage: Backup,
                   walker_index: int) -> Tuple[np.ndarray, float, int, bool, int]:
    """
    Update the position of a walker by picking a move in the list of available moves and accepting the proposed new
        position based on the new cost.

    :param fun: a <d> dimensional (noisy) function to minimize.
    :param x: current position vector for the walker to update.
    :param cost: cost evaluated at last position vector.
    :param current_n: current required number of evaluations.
    :param last_n: number of evaluations required when the last position vector was accepted.
    :param args: a tuple of arguments to pass to the function to minimize.
    :param list_moves: a list of available moves to pick at random.
    :param list_probabilities: a list of corresponding probabilities of picking the move.
    :param iteration: the current iteration number.
    :param max_iter: the maximum number of iterations.
    :param temperature: the current temperature.
    :param complementary_set: the set of position vectors from walkers other than the one to update.
    :param walker_index: the index of the walker to update.

    :return: the updated position vector, cost, number of evaluations and whether the move was accepted.
    """
    # pick a move at random from available moves
    move = np.random.choice(list_moves, p=list_probabilities)

    state = State(complementary_set=complementary_set, iteration=iteration, max_iter=max_iter)

    # generate a new proposal as a neighbor of x and get its cost
    proposed_x = move.get_proposal(x, state)
    previous_evaluations = backup_storage.get_previous_evaluations(proposed_x)
    proposed_cost = get_mean_cost(fun, proposed_x, current_n, args, previous_evaluations)
    backup_storage.save(proposed_x, (current_n, proposed_cost))

    # accept move
    if acceptance_log_probability(cost * current_n / last_n, proposed_cost, temperature) > np.log(np.random.random()):
        return proposed_x, proposed_cost, current_n, True, walker_index

    # reject move
    return x, cost, last_n, False, walker_index


def _vectorized_update_walker(fun: Callable[[np.ndarray, Any], List[float]],
                              x: np.ndarray,
                              costs: Sequence[float],
                              current_n: int,
                              last_ns: Sequence[int],
                              args: Tuple,
                              list_moves: List[Move],
                              list_probabilities: List[float],
                              iteration: int,
                              max_iter: int,
                              temperature: float,
                              backup_storage: Backup) -> Iterator[Tuple[np.ndarray, float, int, bool, int]]:
    """
    Update the positions of a set of walkers using a vectorized cost function, by picking a move in the list of
    available moves and accepting the proposed new position based on the new cost.

    :param fun: a <d> dimensional vectorized (noisy) function to minimize.
    :param x: current position vectors for the walkers to update of shape (nb_walkers,).
    :param costs: set of costs evaluated at last position vectors of shape (nb_walkers,).
    :param current_n: current required number of evaluations.
    :param last_ns: set of number of evaluations required when the last position vectors were accepted of shape
        (nb_walkers,).
    :param args: a tuple of arguments to pass to the function to minimize.
    :param list_moves: a list of available moves to pick at random.
    :param list_probabilities: a list of corresponding probabilities of picking the move.
    :param iteration: the current iteration number.
    :param max_iter: the maximum number of iterations.
    :param temperature: the current temperature.

    :return: an iterator over the updated position vectors, costs, number of evaluations and whether the move were
        accepted.
    """
    # pick a move at random from available moves
    moves = np.random.choice(list_moves, size=len(x), p=list_probabilities)

    states = [State(complementary_set=np.delete(x, walker_index), iteration=iteration, max_iter=max_iter)
              for walker_index in range(len(x))]

    # generate a new proposal as a neighbor of x and get its cost
    proposed_positions = np.array([move.get_proposal(x[walker_index], state)
                                   for walker_index, (move, state) in enumerate(zip(moves, states))])
    previous_evaluations = [backup_storage.get_previous_evaluations(proposed_positions[walker_index])
                            for walker_index, move in enumerate(moves)]
    proposed_costs = get_vectorized_mean_cost(fun, proposed_positions, current_n, args, previous_evaluations)
    for walker_index, move in enumerate(moves):
        backup_storage.save(proposed_positions[walker_index], (current_n, proposed_costs[walker_index]))

    results = ((proposed_positions[walker_index], proposed_costs[walker_index], current_n, True, walker_index)
               if acceptance_log_probability(costs[walker_index] * current_n / last_ns[walker_index],
                                             proposed_costs[walker_index],
                                             temperature) > np.log(np.random.random()) else
               (x[walker_index], costs[walker_index], last_ns[walker_index], False, walker_index)
               for walker_index in range(len(x)))

    return results


class Executor(ABC):
    """
    Base class for function executors.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @abstractmethod
    def map(self,
            fn: Callable,
            *iterables: Iterable,
            **kwargs) -> Iterator:
        """
        Returns an iterator equivalent to map(fn, *iter) for iter in iterables.

        :param fn: a function to evaluate.
        :param iterables: a sequence of iterables to pass to <fn>.

        :return: an iterator over map(fn, *iter).
        """
        pass


class LinearExecutor(Executor):
    """
    Linear executor for calling <fn> with parameters passed in <iterables> in a linear fashion.
    """

    def map(self,
            fn: Callable,
            *iterables: Iterable,
            **kwargs) -> Iterator:
        """
        Returns an iterator equivalent to map(fn, *iter) for iter in iterables.

        :param fn: a function to evaluate.
        :param iterables: a sequence of iterables to pass to <fn>.

        :return: an iterator over map(fn, *iter).
        """
        complementary_sets = [np.delete(kwargs['positions'], walker_index)
                              for walker_index in range(len(kwargs['positions']))]

        return map(_update_walker, repeat(fn), *iterables, complementary_sets, repeat(kwargs['backup']),
                   range(len(kwargs['positions'])))


class VectorizedExecutor(Executor):
    """
    Vectorized executor for calling <fn> with all parameters defined in <iterables> at once. This requires <fn> to be
    a vectorized function.
    """

    def map(self,
            fn: Callable,
            *iterables: Iterable,
            **kwargs) -> Iterator:
        """
        Returns an iterator equivalent to map(fn, *iter) for iter in iterables.

        :param fn: a vectorized function to evaluate.
        :param iterables: a sequence of iterables to pass to <fn>.

        :return: an iterator over map(fn, *iter).
        """
        return _vectorized_update_walker(fn,
                                         x=cast(np.ndarray, iterables[0]),                              # type: ignore
                                         costs=cast(List[float], iterables[1]),                         # type: ignore
                                         current_n=cast(int, next(iterables[2])),                       # type: ignore
                                         last_ns=cast(List[int], iterables[3]),                         # type: ignore
                                         args=cast(Tuple, next(iterables[4])),                          # type: ignore
                                         list_moves=cast(List[Move], next(iterables[5])),               # type: ignore
                                         list_probabilities=cast(List[float], next(iterables[6])),      # type: ignore
                                         iteration=cast(int, next(iterables[7])),                       # type: ignore
                                         max_iter=cast(int, next(iterables[8])),                        # type: ignore
                                         temperature=cast(float, next(iterables[9])),                   # type: ignore
                                         backup_storage=kwargs['backup'])


class ParallelExecutor(ProcessPoolExecutor):
    """
    Executor for calling <fn> in parallel one multiple cores.
    """

    def __init__(self, max_workers: int):
        super(ParallelExecutor, self).__init__(max_workers=max_workers)
        self.manager = BackupManager()
        self.manager.start()
        self.backup = self.manager.Backup()                                                     # type: ignore

    def map(self,                                                                               # type: ignore
            fn: Callable,
            *iterables: Iterable,
            timeout=None,
            chunksize=1,
            **kwargs) -> Iterator:
        """
        Returns an iterator equivalent to map(fn, *iter) for iter in iterables.

        :param fn: a function to evaluate.
        :param iterables: a sequence of iterables to pass to <fn>.
        :param timeout:
        :param chunksize:

        :return: an iterator over map(fn, *iter).
        """
        complementary_sets = [np.delete(kwargs['positions'], walker_index)
                              for walker_index in range(len(kwargs['positions']))]

        return super().map(_update_walker,
                           *(repeat(fn), *iterables, complementary_sets, repeat(self.backup),   # type: ignore
                             range(len(kwargs['positions']))),
                           timeout=timeout,
                           chunksize=chunksize)