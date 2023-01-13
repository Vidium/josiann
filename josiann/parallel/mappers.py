# coding: utf-8
# Created on 16/06/2022 14:50
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import Sequence

from josiann.compute import acceptance_log_probability
from josiann.parallel.compute import get_vectorized_mean_cost
from josiann.moves.parallel.base import ParallelMove
from josiann.backup.parallel.backup import ParallelBackup

import josiann.typing as jot


# ====================================================
# code
def _vectorized_update_walker(
    fun: jot.VECT_FUN_TYPE,
    x: npt.NDArray[jot.DType],
    converged: npt.NDArray[np.bool_],
    costs: Sequence[float],
    current_n: int,
    last_ns: Sequence[int],
    parallel_args: Sequence[npt.NDArray[Any]] | None,
    args: tuple[Any, ...],
    list_moves: list[ParallelMove],
    list_probabilities: list[float],
    temperature: float,
    backup_storage: ParallelBackup,
) -> Iterator[tuple[npt.NDArray[jot.DType] | float, float, bool, int]]:
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
    move = np.random.choice(list_moves, p=list_probabilities)  # type: ignore[arg-type]
    proposed_positions = move.get_proposal(x[~converged], None)

    previous_evaluations = backup_storage.get_previous_evaluations(proposed_positions)

    proposed_costs = get_vectorized_mean_cost(
        fun,
        proposed_positions,
        current_n,
        converged,
        parallel_args,
        args,
        previous_evaluations,
    )

    backup_storage.save(
        proposed_positions, [(current_n, cost) for cost in proposed_costs]
    )

    evaluation_index = 0

    for index, has_converged in enumerate(converged):
        if has_converged:
            yield np.nan, np.nan, True, index

        else:
            position = proposed_positions[evaluation_index]
            proposed_cost = proposed_costs[evaluation_index]

            evaluation_index += 1

            if acceptance_log_probability(
                costs[index] * current_n / last_ns[index], proposed_cost, temperature
            ) > np.log(np.random.random()):
                accepted = True

            else:
                accepted = False

            yield position, proposed_cost, accepted, index


def vectorized_execution(
    fun: jot.VECT_FUN_TYPE, *iterables: Iterable[Any], **kwargs: Any
) -> Iterator[tuple[npt.NDArray[jot.DType] | float, float, bool, int]]:
    """
    Vectorized executor for calling <fn> with all parameters defined in <iterables> at once. This requires <fn> to be
    a vectorized function.

    Args:
        fun: a vectorized function to evaluate.
        *iterables: a sequence of iterables to pass to <fun>.
        **kwargs: additional parameters.

    Returns:
        An iterator over map(fn, *iter).
    """
    return _vectorized_update_walker(
        fun,
        x=cast(npt.NDArray[jot.DType], iterables[0]),
        converged=cast(npt.NDArray[np.bool_], iterables[1]),
        costs=kwargs["costs"],
        current_n=kwargs["current_n"],
        last_ns=kwargs["last_ns"],
        parallel_args=kwargs["parallel_args"],
        args=kwargs["args"],
        list_moves=kwargs["list_moves"],
        list_probabilities=kwargs["list_probabilities"],
        temperature=kwargs["temperature"],
        backup_storage=kwargs["backup"],
    )
