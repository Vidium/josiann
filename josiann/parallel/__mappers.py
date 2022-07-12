# coding: utf-8
# Created on 16/06/2022 14:50
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

from typing import Callable, Iterable, Iterator, Sequence, Any, cast

from josiann.utils import acceptance_log_probability
from josiann.__mappers import Executor
from .utils import get_vectorized_mean_cost
from .moves import ParallelMove
from .__backup import Backup


# ====================================================
# code
def _vectorized_update_walker(fun: Callable[[np.ndarray, Any], list[float]],
                              x: np.ndarray,
                              converged: np.ndarray,
                              costs: Sequence[float],
                              current_n: int,
                              last_ns: Sequence[int],
                              parallel_args: Sequence[np.ndarray] | None,
                              args: tuple,
                              list_moves: list[ParallelMove],
                              list_probabilities: list[float],
                              temperature: float,
                              backup_storage: Backup) -> Iterator[tuple[np.ndarray, float, int, bool, int]]:
    """
    Update the positions of a set of walkers using a vectorized cost function, by picking a move in the list of
    available moves and accepting the proposed new position based on the new cost.

    Args:
        fun: a <d> dimensional vectorized (noisy) function to minimize.
        x: current position vectors for the walkers to update of shape (nb_walkers,).
        converged: a vector indicating which walkers have already converged.
        costs: set of costs evaluated at last position vectors of shape (nb_walkers,).
        current_n: current required number of evaluations.
        last_ns: set of number of evaluations required when the last position vectors were accepted of shape
        (nb_walkers,).
        args: a tuple of arguments to pass to the function to minimize.
        list_moves: a list of available moves to pick at random.
        list_probabilities: a list of corresponding probabilities of picking the move.
        temperature: the current temperature.
        backup_storage: a Backup object for storing previously computed positions.

    Returns:
        An iterator over the updated position vectors, costs, number of evaluations and whether the move were
        accepted.
    """
    # generate a new proposal as a neighbor of x and get its cost
    move = np.random.choice(list_moves, p=list_probabilities)
    proposed_positions = move.get_proposal(x[~converged], None)

    previous_evaluations = backup_storage.get_previous_evaluations(proposed_positions)

    proposed_costs = get_vectorized_mean_cost(
        fun,
        proposed_positions,
        current_n,
        converged,
        parallel_args,
        args,
        previous_evaluations
    )

    backup_storage.save(proposed_positions, [(current_n, cost) for cost in proposed_costs])

    evaluation_index = 0

    for index, has_converged in enumerate(converged):
        if has_converged:
            yield np.nan, np.nan, np.nan, True, index

        else:
            position = proposed_positions[evaluation_index]
            proposed_cost = proposed_costs[evaluation_index]

            evaluation_index += 1

            if acceptance_log_probability(costs[index] * current_n / last_ns[index],
                                          proposed_cost,
                                          temperature) > np.log(np.random.random()):
                yield position, proposed_cost, current_n, True, index

            else:
                yield position, costs[index], last_ns[index], False, index


class VectorizedExecutor(Executor):
    """
    Vectorized executor for calling <fun> . This requires <fun> to be a vectorized function.
    """

    def map(self,
            fun: Callable,
            *iterables: Iterable,
            **kwargs) -> Iterator:
        """
        Returns an iterator on the results of <fun> applied to each position vector in x.

        Args:
            fun: a vectorized function to evaluate.
            iterables: a sequence of iterables to pass to <fun>.

        Returns:
            An iterator over map(fn, *iter).
        """
        return _vectorized_update_walker(fun,
                                         x=cast(np.ndarray, iterables[0]),
                                         converged=cast(np.ndarray, iterables[1]),
                                         costs=kwargs['costs'],
                                         current_n=kwargs['current_n'],
                                         last_ns=kwargs['last_ns'],
                                         parallel_args=kwargs['parallel_args'],
                                         args=kwargs['args'],
                                         list_moves=kwargs['list_moves'],
                                         list_probabilities=kwargs['list_probabilities'],
                                         temperature=kwargs['temperature'],
                                         backup_storage=kwargs['backup'])
