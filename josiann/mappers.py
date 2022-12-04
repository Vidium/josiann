# coding: utf-8
# Created on 03/08/2021 12:21
# Author : matteo

"""
Defines executors for updating the position vectors by calling moves and computing the costs:
    LinearExecutor: sequential updates
    VectorizedExecutor: updates all at once in matrix of positions
    ParallelExecutor: parallel updates
"""

# ====================================================
# imports
from typing import Any

import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from typing import Iterator
from typing import Sequence

from josiann.moves import Move
from josiann.moves import State
from josiann.backup import Backup
from josiann.backup import BackupManager
from josiann.compute import get_mean_cost
from josiann.compute import acceptance_log_probability
from josiann.typing import Execution
from josiann.typing import FUN_TYPE
from josiann.typing import SA_UPDATE


# ====================================================
# code
def _update_walker(fun: FUN_TYPE,
                   x: np.ndarray,
                   cost: float,
                   current_n: int,
                   last_n: int,
                   args: tuple,
                   list_moves: list[Move],
                   list_probabilities: list[float],
                   iteration: int,
                   max_iter: int,
                   temperature: float,
                   backup_storage: Backup,
                   complementary_set: np.ndarray,
                   walker_index: int) -> SA_UPDATE:
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
    move = np.random.choice(list_moves, p=list_probabilities)                   # type: ignore[arg-type]

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


def linear_execution(fn: FUN_TYPE,
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
    Linear executor for calling <fn> with parameters passed in <iterables> in a linear fashion.
    Equivalent to map(fn, *iter) for iter in iterables.

    Args:
        fn: a function to evaluate.
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

    Returns:
        An iterator over map(fn, *iter).
    """
    complementary_sets = [np.delete(x.copy(), walker_index) for walker_index in range(len(x))]

    return map(_update_walker,
               repeat(fn),
               x,
               costs,
               repeat(current_n),
               last_ns,
               repeat(args),
               repeat(list_moves),
               repeat(list_probabilities),
               repeat(iteration),
               repeat(max_iter),
               repeat(temperature),
               repeat(backup_storage),
               complementary_sets,
               range(len(x)))


def parallel_execution(max_workers: int) -> Execution:
    """


    Args:
        max_workers: max number of parallel workers.

    Returns:
        An executor for calling <fn> in parallel on multiple cores.
    """
    executor = ProcessPoolExecutor(max_workers=max_workers)

    manager = BackupManager()
    manager.start()
    backup = manager.SequentialBackup()                                 # type: ignore[attr-defined]

    def call(fn: FUN_TYPE,
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
        Executor for calling <fn> in parallel on multiple cores.

        Args:
            fn: a function to evaluate.
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
                timeout: The maximum number of seconds to wait. If None, then there is no limit on the wait time.
                chunksize: The size of the chunks the iterable will be broken into before being passed to a child
                    process. This argument is only used by ProcessPoolExecutor; it is ignored by ThreadPoolExecutor.

        Returns:
            An iterator over map(fn, *iter).
        """
        complementary_sets = [np.delete(kwargs['positions'], walker_index)
                              for walker_index in range(len(kwargs['positions']))]

        return executor.map(_update_walker,
                            repeat(fn),
                            x,
                            costs,
                            repeat(current_n),
                            last_ns,
                            repeat(args),
                            repeat(list_moves),
                            repeat(list_probabilities),
                            repeat(iteration),
                            repeat(max_iter),
                            repeat(temperature),
                            repeat(backup_storage),
                            complementary_sets,
                            repeat(backup),
                            range(len(x)),
                            timeout=kwargs['timeout'],
                            chunksize=kwargs['chunksize'])

    return call
