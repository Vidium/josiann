# coding: utf-8
# Created on 26/07/2021 12:08
# Author : matteo

# ====================================================
# imports
import collections
import numpy as np

from typing import Union, Sequence, Tuple, List, Callable, Any

from .moves import Move


# ====================================================
# code
def parse_moves(moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]]) -> Tuple[List[float], List[Move]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    :param moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
        sequence of tuples with format (probability: float, Move).

    :return: the list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.Sequence) or isinstance(moves, str):
        if isinstance(moves, Move):
            return [1.0], [moves]

        raise ValueError(f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a 'Move', "
                         f"a sequence of 'Move' or a sequence of tuples '(probability: float, Move)'.")

    parsed_probabilites = []
    parsed_moves = []

    for move in moves:
        if isinstance(move, Move):
            parsed_probabilites.append(1.0)
            parsed_moves.append(move)

        elif isinstance(move, tuple):
            if len(move) == 2 and isinstance(move[0], float) and isinstance(move[1], Move):
                parsed_probabilites.append(move[0])
                parsed_moves.append(move[1])

            else:
                raise ValueError(f"Invalid format for tuple '{move}', expected '(probability: float, Move)'.")

        else:
            raise ValueError(f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                             f"defining a move, expected a 'Move' or tuple '(probability: float, Move)'.")

    if sum(parsed_probabilites) != 1:
        _sum = sum(parsed_probabilites)
        parsed_probabilites = [proba / _sum for proba in parsed_probabilites]

    return parsed_probabilites, parsed_moves


def get_mean_cost(fun: Callable[[np.ndarray, Any], float], x: np.ndarray, n: int, *args) -> float:
    """
    Get the mean of <n> function evaluations for vector of values <x>.

    :param fun: a function to evaluate.
    :param x: a vector of values.
    :param n: the number of evaluations to compute.

    :return: the mean of function evaluations at x.
    """
    return float(np.mean([fun(x, *args) for _ in range(n)]))
