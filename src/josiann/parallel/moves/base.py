# Created on 16/06/2022 11:18
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import collections.abc
from abc import ABC
from collections.abc import Sequence

from josiann.moves.base import Move


# ====================================================
# code
# Moves for parallel problems
class ParallelMove(Move, ABC):
    """
    Base class for building moves that solve optimization problems in parallel.
    """


# functions
def parse_moves(
    moves: ParallelMove | Sequence[ParallelMove] | Sequence[tuple[float, ParallelMove]],
) -> tuple[list[float], list[ParallelMove]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    Args:
        moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
            sequence of tuples with format (probability: float, Move).

    Returns:
        The list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.abc.Sequence) or isinstance(moves, str):
        if isinstance(moves, ParallelMove):
            return [1.0], [moves]

        raise ValueError(
            f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a "
            f"'ParallelMove', a sequence of 'ParallelMove's or a sequence of tuples "
            f"'(probability: float, 'ParallelMove')'."
        )

    parsed_probabilities = []
    parsed_moves = []

    for move in moves:
        if isinstance(move, ParallelMove):
            parsed_probabilities.append(1.0)
            parsed_moves.append(move)

        elif isinstance(move, tuple):
            if len(move) == 2 and isinstance(move[0], float) and isinstance(move[1], ParallelMove):
                parsed_probabilities.append(move[0])
                parsed_moves.append(move[1])

            else:
                raise ValueError(f"Invalid format for tuple '{move}', expected '(probability: float, Move)'.")

        else:
            raise ValueError(
                f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                f"defining a move, expected a 'ParallelMove' or tuple "
                f"'(probability: float, 'ParallelMove')'."
            )

    if sum(parsed_probabilities) != 1:
        _sum = sum(parsed_probabilities)
        parsed_probabilities = [proba / _sum for proba in parsed_probabilities]

    return parsed_probabilities, parsed_moves
