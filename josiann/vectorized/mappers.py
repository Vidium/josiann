# coding: utf-8
# Created on 03/12/2022 17:12
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import Any
from typing import Sequence
from typing import Iterator

from josiann.moves import Move
from josiann.moves import State
from josiann.backup import Backup
from josiann.compute import acceptance_log_probability
from josiann.typing import SA_UPDATE
from josiann.typing import VECT_FUN_TYPE
from josiann.vectorized.compute import get_evaluation_vectorized_mean_cost
from josiann.vectorized.compute import get_walker_vectorized_mean_cost


# ====================================================
# code
def _get_exploration_plan(acceptance: float,
                          nb_slots: int,
                          x: np.ndarray,
                          list_moves: Sequence['Move'],
                          list_probabilities: list[float],
                          state: State,
                          plan: list[np.ndarray]) -> tuple[int, int, list[np.ndarray]]:
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
    used = max(int(np.ceil((1-acceptance) * nb_slots)), 1)
    left = nb_slots - used

    for _ in range(used):
        move = np.random.choice(list_moves, p=list_probabilities)               # type: ignore[arg-type]
        proposal = move.get_proposal(x, state)
        plan.append(proposal)

        if left > 0:
            used_after, *_ = _get_exploration_plan(acceptance, int(np.ceil(left / used)), proposal,
                                                   list_moves, list_probabilities,
                                                   State(state.complementary_set, state.iteration + 1, state.max_iter),
                                                   plan)
            left -= used_after

    return used, left, plan


def vectorized_execution(fn: VECT_FUN_TYPE,
                         x: np.ndarray,
                         costs: np.ndarray,
                         current_n: int,
                         last_ns: np.ndarray,
                         args: tuple[Any, ...],
                         list_moves: Sequence[Move],
                         list_probabilities: list[float],
                         iteration: int,
                         max_iter: int,
                         temperature: float,
                         backup_storage: Backup,
                         **kwargs: Any) -> Iterator[SA_UPDATE]:
    """
    Update the positions of a set of walkers using a vectorized cost function, by picking a move in the list of
    available moves and accepting the proposed new position based on the new cost.

    Args:
        fn: a <d> dimensional vectorized (noisy) function to minimize.
        x: current position vectors for the walkers to update of shape (nb_walkers,).
        costs: set of costs evaluated at last position vectors of shape (nb_walkers,).
        current_n: current required number of evaluations.
        last_ns: set of number of evaluations required when the last position vectors were accepted of shape
            (nb_walkers,).
        args: a tuple of arguments to pass to the function to minimize.
        list_moves: a list of available moves to pick at random.
        list_probabilities: a list of corresponding probabilities of picking the move.
        iteration: the current iteration number.
        max_iter: the maximum number of iterations.
        temperature: the current temperature.
        backup_storage: a Backup object for storing previously computed positions.
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
    # return _vectorized_update_walker(fn,
    #                                  x=x,
    #                                  costs=costs,
    #                                  current_n=current_n,
    #                                  last_ns=last_ns,
    #                                  args=args,
    #                                  list_moves=list_moves,
    #                                  list_probabilities=list_probabilities,
    #                                  iteration=iteration,
    #                                  max_iter=max_iter,
    #                                  temperature=temperature,
    #                                  acceptance=acceptance,
    #                                  backup_storage=backup_storage,
    #                                  nb_slots=kwargs['nb_slots'],
    #                                  vectorized_on_evaluations=kwargs['vectorized_on_evaluations'],
    #                                  vectorized_skip_marker=kwargs['vectorized_skip_marker'])

    states = [State(complementary_set=np.delete(x, walker_index), iteration=iteration, max_iter=max_iter)
              for walker_index in range(len(x))]

    # generate a new proposal as a neighbor of x and get its cost
    proposed_positions = np.array([_get_exploration_plan(kwargs['acceptance'],
                                                         kwargs['nb_slots'][walker_index],
                                                         x[walker_index],
                                                         list_moves,
                                                         list_probabilities,
                                                         states[walker_index], [])[2]
                                   for walker_index in range(len(x))]).reshape(sum(kwargs['nb_slots']), x.shape[1])

    if kwargs['vectorized_on_evaluations']:
        unique_proposed_positions = np.unique(proposed_positions, axis=0)

        previous_evaluations = [backup_storage.get_previous_evaluations(unique_proposed_positions[index])
                                for index in range(len(unique_proposed_positions))]

        unique_proposed_costs = get_evaluation_vectorized_mean_cost(fn, unique_proposed_positions, current_n, args,
                                                                    previous_evaluations)

        proposed_costs = np.zeros((len(proposed_positions)))
        for i, cost in enumerate(unique_proposed_costs):
            proposed_costs[np.all(proposed_positions == unique_proposed_positions[i], axis=1)] = cost

            backup_storage.save(unique_proposed_positions[i], (current_n, cost))

    else:
        previous_evaluations = [backup_storage.get_previous_evaluations(proposed_positions[index])
                                for index in range(len(proposed_positions))]

        proposed_costs = np.array(get_walker_vectorized_mean_cost(fn,
                                                                  proposed_positions,
                                                                  current_n,
                                                                  args,
                                                                  previous_evaluations,
                                                                  kwargs['vectorized_skip_marker']))

        for i, cost in enumerate(proposed_costs):
            backup_storage.save(proposed_positions[i], (current_n, cost))

    results = []

    for walker_index, walker_position in enumerate(x):
        best_index = np.argmin(proposed_costs[walker_index * kwargs['nb_slots'][walker_index]:
                                              (walker_index + 1) * kwargs['nb_slots'][walker_index]])

        if acceptance_log_probability(costs[walker_index] * current_n / last_ns[walker_index],
                                      proposed_costs[walker_index * kwargs['nb_slots'][walker_index] + best_index],
                                      temperature) > np.log(np.random.random()):
            results.append((proposed_positions[walker_index * kwargs['nb_slots'][walker_index] + best_index],
                            proposed_costs[walker_index * kwargs['nb_slots'][walker_index] + best_index],
                            True,
                            walker_index))

        else:
            results.append((walker_position, costs[walker_index], False, walker_index))

    return iter(results)
