# ====================================================
# imports
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

import josiann.typing as jot
from josiann.algorithms.sequential.base.compute import get_mean_cost
from josiann.backup.backup import SequentialBackup
from josiann.compute import acceptance_log_probability
from josiann.moves.base import Move, State


# ====================================================
# code
def _update_walker(
    fun: jot.FUN_TYPE[...],
    x: NDArray[jot.DType],
    cost: float,
    current_n: int,
    last_n: int,
    args: tuple[Any, ...],
    list_moves: list[Move],
    list_probabilities: list[float],
    iteration: int,
    max_iter: int,
    temperature: float,
    backup_storage: SequentialBackup,
    complementary_set: NDArray[jot.DType],
    walker_index: int,
) -> tuple[NDArray[jot.DType], float, bool, int]:
    """
    Update the position of a walker by picking a move in the list of available moves and accepting the proposed new
        position based on the new cost.

    Args:
        fun: a <d> dimensional (noisy) function to minimize.
        x: current position vector for the walker to update.
        cost: cost evaluated at last position vector.
        current_n: current required number of evaluations.
        last_n: number of evaluations required when the last position vector was accepted.
        args: a tuple of arguments to pass to the function to minimize.
        list_moves: a list of available moves to pick at random.
        list_probabilities: a list of corresponding probabilities of picking the move.
        iteration: the current iteration number.
        max_iter: the maximum number of iterations.
        temperature: the current temperature.
        complementary_set: the set of position vectors from walkers other than the one to update.
        backup_storage: a Backup object for storing previously computed positions.
        walker_index: the index of the walker to update.

    Returns:
        The updated position vector, cost, number of evaluations and whether the move was accepted.
    """
    # pick a move at random from available moves
    move: Move = np.random.choice(list_moves, p=list_probabilities)  # type: ignore[arg-type]

    state = State(complementary_set=complementary_set, iteration=iteration, max_iter=max_iter)

    # generate a new proposal as a neighbor of x and get its cost
    proposed_x = move.get_proposal(x, state)
    previous_evaluations = backup_storage.get_previous_evaluations(proposed_x)
    proposed_cost = get_mean_cost(fun, proposed_x, current_n, args, previous_evaluations)
    backup_storage.save(proposed_x, (current_n, proposed_cost))

    # accept move
    if acceptance_log_probability(cost * current_n / last_n, proposed_cost, temperature) > np.log(np.random.random()):
        return proposed_x, proposed_cost, True, walker_index

    # reject move
    return x, cost, False, walker_index
