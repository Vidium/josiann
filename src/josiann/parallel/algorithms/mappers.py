# ====================================================
# imports
from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from enum import Enum, auto
from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.backup.parallel.backup import ParallelBackup
from josiann.compute import acceptance_log_probability
from josiann.parallel.algorithms.compute import get_vectorized_mean_cost
from josiann.parallel.moves.base import ParallelMove


# ====================================================
# code
class UpdateState(Enum):
    accepted = auto()
    rejected = auto()
    converged = auto()


def _vectorized_update_walker(
    fun: jot.PARALLEL_FUN_TYPE[...],
    x: npt.NDArray[jot.DType],
    converged: npt.NDArray[np.bool_],
    costs: Sequence[float],
    current_n: int,
    last_ns: Sequence[int],
    parallel_args: tuple[npt.NDArray[Any]],
    args: tuple[Any, ...],
    list_moves: list[ParallelMove],
    list_probabilities: list[float],
    temperature: float,
    backup_storage: ParallelBackup,
) -> Generator[tuple[npt.NDArray[jot.DType] | float, float, UpdateState], None, None]:
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

    proposed_positions = np.zeros_like(x)
    proposed_positions[~converged] = move.get_proposal(x[~converged], None)

    previous_evaluations = [
        (current_n, 0.0) if converged[i] else p
        for i, p in enumerate(backup_storage.get_previous_evaluations(proposed_positions))
    ]

    proposed_costs = get_vectorized_mean_cost(
        fun,
        proposed_positions,
        current_n,
        parallel_args,
        args,
        previous_evaluations,
    )

    backup_storage.save(
        proposed_positions[~converged],
        [(current_n, cost) for cost in proposed_costs[~converged]],
    )

    for has_converged, cost, last_n, proposed_cost, proposed_position in zip(
        converged, costs, last_ns, proposed_costs, proposed_positions
    ):
        if has_converged:
            yield np.nan, np.nan, UpdateState.converged

        else:
            if acceptance_log_probability(cost * current_n / last_n, proposed_cost, temperature) > np.log(
                np.random.random()
            ):
                yield proposed_position, proposed_cost, UpdateState.accepted

            else:
                yield proposed_position, proposed_cost, UpdateState.rejected


def vectorized_execution(
    fun: jot.PARALLEL_FUN_TYPE[...],
    x: npt.NDArray[jot.DType],
    converged: npt.NDArray[np.bool_],
    **kwargs: Any,
) -> Iterator[tuple[npt.NDArray[jot.DType] | float, float, UpdateState]]:
    """
    Vectorized executor for calling <fn> on all position vectors in <x> at once. This requires <fn> to be
    a vectorized function.

    Args:
        fun: a vectorized function to evaluate.
        x: position vectors.
        converged: array of booleans values indicating if a problem as already converged.
        **kwargs: additional parameters.

    Returns:
        An iterator over map(fn, *iter).
    """
    return _vectorized_update_walker(
        fun,
        x=x,
        converged=converged,
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
