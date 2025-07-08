# Created on 13/01/2023 09:51
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import collections.abc
from collections.abc import Sequence

from josiann.moves.base import Move
from josiann.moves.ensemble import EnsembleMove


# ====================================================
# code
def parse_moves(
    moves: Move | Sequence[Move] | Sequence[tuple[float, Move]], nb_walkers: int
) -> tuple[list[float], list[Move]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    Args:
        moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
            sequence of tuples with format (probability: float, Move).
        nb_walkers: the number of parallel walkers in the ensemble.

    Returns:
        The list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.abc.Sequence) or isinstance(moves, str):
        if isinstance(moves, Move):
            if issubclass(type(moves), EnsembleMove) and nb_walkers < 2:
                raise ValueError("Ensemble moves require at least 2 walkers to be used.")

            return [1.0], [moves]

        raise ValueError(
            f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a "
            f"'Move', a sequence of 'Move's or a sequence of tuples "
            f"'(probability: float, 'Move')'."
        )

    parsed_probabilities = []
    parsed_moves = []

    for move in moves:
        if isinstance(move, Move):
            if issubclass(type(move), EnsembleMove) and nb_walkers < 2:
                raise ValueError("Ensemble moves require at least 2 walkers to be used.")

            parsed_probabilities.append(1.0)
            parsed_moves.append(move)

        elif isinstance(move, tuple):
            if len(move) == 2 and isinstance(move[0], float) and isinstance(move[1], Move):
                if issubclass(type(move[1]), EnsembleMove) and nb_walkers < 2:
                    raise ValueError("Ensemble moves require at least 2 walkers to be used.")

                parsed_probabilities.append(move[0])
                parsed_moves.append(move[1])

            else:
                raise ValueError(f"Invalid format for tuple '{move}', expected '(probability: float, Move)'.")

        else:
            raise ValueError(
                f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                f"defining a move, expected a 'Move' or tuple '(probability: float, 'Move')'."
            )

    if sum(parsed_probabilities) != 1:
        _sum = sum(parsed_probabilities)
        parsed_probabilities = [proba / _sum for proba in parsed_probabilities]

    return parsed_probabilities, parsed_moves
