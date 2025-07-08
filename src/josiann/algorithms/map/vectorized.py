# ====================================================
# imports
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.algorithms.sequential.vectorized.compute import (
    get_evaluation_vectorized_mean_cost,
    get_walker_vectorized_mean_cost,
)
from josiann.compute import acceptance_log_probability
from josiann.moves.base import Move, State
from josiann.storage.parameters import SAParameters


# ====================================================
# code
def _get_exploration_plan(
    acceptance: float,
    nb_slots: int,
    x: npt.NDArray[jot.DType],
    list_moves: Sequence[Move],
    list_probabilities: list[float],
    state: State,
    plan: list[npt.NDArray[jot.DType]],
) -> tuple[int, int, list[npt.NDArray[jot.DType]]]:
    """
    Define an exploration plan : a set of moves (of size <nb_slots>) to take from an initial position vector <x>.
    For low acceptance values, the moves are mostly picked around the original position vector in a star pattern.
    For high acceptance values, since the probability of a move being accepted is higher, moves are taken more
        sequentially.

    Args:
        acceptance: the current proportion of accepted moves.
        nb_slots: the size of the exploration plan (number of moves to take).
        x: the current position vector of shape (ndim,)
        list_moves: the list of possible moves from which to choose.
        list_probabilities: the associated probabilities.
        state: the current state of the SA algorithm.
        plan: the exploration plan, as a list of position vectors.

    Returns:
        The number of used slots, the number of slots left, the exploration plan.
    """
    used = max(int(np.ceil((1 - acceptance) * nb_slots)), 1)
    left = nb_slots - used

    for _ in range(used):
        move = np.random.choice(list_moves, p=list_probabilities)  # type: ignore[arg-type]
        proposal = move.get_proposal(x, state)
        plan.append(proposal)

        if left > 0:
            used_after, *_ = _get_exploration_plan(
                acceptance,
                int(np.ceil(left / used)),
                proposal,
                list_moves,
                list_probabilities,
                State(state.complementary_set, state.iteration + 1, state.max_iter),
                plan,
            )
            left -= used_after

    return used, left, plan


def vectorized_execution(
    params: SAParameters,
    x: npt.NDArray[jot.DType],
    costs: npt.NDArray[np.float64],
    current_n: int,
    last_ns: npt.NDArray[np.int64],
    iteration: int,
    temperature: float,
    **kwargs: Any,
) -> Iterator[tuple[npt.NDArray[jot.DType], float, bool, int]]:
    """
    Update the positions of a set of walkers using a vectorized cost function, by picking a move in the list of
    available moves and accepting the proposed new position based on the new cost.

    Args:
        params: parameters of an SA run.
        x: current position vectors for the walkers to update of shape (nb_walkers,).
        costs: set of costs evaluated at last position vectors of shape (nb_walkers,).
        current_n: current required number of evaluations.
        last_ns: set of number of evaluations required when the last position vectors were accepted of shape
            (nb_walkers,).
        iteration: the current iteration number.
        temperature: the current temperature.
        kwargs:
            acceptance: the current acceptance fraction.
            nb_slots: the list of slots per walker.
            vectorized_on_evaluations: vectorize <fun> calls on evaluations (or walkers) ?
            vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
                evaluation for a particular position vector can be skipped.

    Returns:
        An iterator over the updated position vectors, costs, number of evaluations and whether the move were
        accepted.
    """
    states = [
        State(
            complementary_set=np.delete(x, walker_index),
            iteration=iteration,
            max_iter=params.base.max_iter,
        )
        for walker_index in range(len(x))
    ]

    # generate a new proposal as a neighbor of x and get its cost
    proposed_positions = np.array(
        [
            _get_exploration_plan(
                kwargs["acceptance"],
                kwargs["nb_slots"][walker_index],
                x[walker_index],
                params.moves.list_moves,
                params.moves.list_probabilities,
                states[walker_index],
                [],
            )[2]
            for walker_index in range(len(x))
        ]
    ).reshape(sum(kwargs["nb_slots"]), x.shape[1])

    if kwargs["vectorized_on_evaluations"]:
        unique_proposed_positions = np.unique(proposed_positions, axis=0)

        previous_evaluations = [
            params.backup.get_previous_evaluations(unique_proposed_positions[index])
            for index in range(len(unique_proposed_positions))
        ]

        unique_proposed_costs = get_evaluation_vectorized_mean_cost(
            params.fun,  # type: ignore[arg-type]
            unique_proposed_positions,
            current_n,
            params.base.args,
            previous_evaluations,
        )

        proposed_costs: npt.NDArray[np.floating[Any]] = np.zeros(len(proposed_positions))
        for i, cost in enumerate(unique_proposed_costs):
            proposed_costs[np.all(proposed_positions == unique_proposed_positions[i], axis=1)] = cost

            params.backup.save(unique_proposed_positions[i], (current_n, cost))

    else:
        previous_evaluations = [
            params.backup.get_previous_evaluations(proposed_positions[index])
            for index in range(len(proposed_positions))
        ]

        proposed_costs = get_walker_vectorized_mean_cost(
            params.fun,  # type: ignore[arg-type]
            proposed_positions,
            current_n,
            params.base.args,
            previous_evaluations,
            kwargs["vectorized_skip_marker"],
        )

        for i, cost in enumerate(proposed_costs):
            params.backup.save(proposed_positions[i], (current_n, cost))

    results = []

    for walker_index, walker_position in enumerate(x):
        best_index = np.argmin(
            proposed_costs[
                walker_index * kwargs["nb_slots"][walker_index] : (walker_index + 1) * kwargs["nb_slots"][walker_index]
            ]
        )

        if acceptance_log_probability(
            costs[walker_index] * current_n / last_ns[walker_index],
            proposed_costs[walker_index * kwargs["nb_slots"][walker_index] + best_index],
            temperature,
        ) > np.log(np.random.random()):
            results.append(
                (
                    proposed_positions[walker_index * kwargs["nb_slots"][walker_index] + best_index],
                    proposed_costs[walker_index * kwargs["nb_slots"][walker_index] + best_index],
                    True,
                    walker_index,
                )
            )

        else:
            results.append((walker_position, costs[walker_index], False, walker_index))

    return iter(results)
