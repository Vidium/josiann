# coding: utf-8
# Created on 16/06/2022 11:18
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import collections.abc
from abc import ABC

from typing import Sequence

from josiann import Move
from josiann.utils import State
from josiann.name_utils import ShapeError


# ====================================================
# code
# Moves for parallel problems
class ParallelMove(Move, ABC):
    """
    Base class for building moves that solve optimization problems in parallel.
    """


class ParallelSetStep(ParallelMove):
    """
    Step within a fixed set of possible values for x. For each dimension, the position immediately before or after x
        will be chosen at random when stepping.

    Args:
        position_set: sets of only possible values for x in each dimension.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 position_set: Sequence[np.ndarray],
                 bounds: Sequence[tuple[float, float]] | None = None):
        super().__init__(bounds=bounds)

        if not all(isinstance(p, (Sequence, np.ndarray)) for p in position_set):
            raise ShapeError("'position_set' parameter should be an array of possible position values of shape "
                             "(dimensions, nb_values) (nb_values can be different for each dimension).")

        self._position_set = [np.sort(p) for p in position_set]
        self._reversed_position_set = [v[::-1] for v in self._position_set]
        self._target_dim = 0
        self._dtype = None

    def __repr__(self):
        return f"[Move] ParallelSetStep : {self._position_set}"

    def set_dtype(self,
                  dtype: np.dtype) -> None:
        """
        Set a data type for the positions generated by this move.

        Args:
            dtype: the data type to set.
        """
        for index, pos_set in enumerate(self._position_set):
            self._position_set[index] = pos_set.astype(dtype)

        for index, pos_set in enumerate(self._reversed_position_set):
            self._reversed_position_set[index] = pos_set.astype(dtype)

        self._dtype = dtype

    def _get_proposal(self,
                      x: np.ndarray,
                      state: State | None) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        new_x = x.copy().astype(self._dtype)

        # loop on vectors in x
        for index in range(len(x)):
            # for each, draw at random if the position increases ...
            if np.random.rand() > 0.5:
                mask = self._position_set[self._target_dim] > x[index, self._target_dim]
                if np.any(mask):
                    new_x[index, self._target_dim] = self._position_set[self._target_dim][np.argmax(mask)]
                # boundary hit : cannot go higher than the highest value --> go down instead
                else:
                    new_x[index, self._target_dim] = self._position_set[self._target_dim][-2]

            # ... or decreases
            else:
                mask = self._reversed_position_set[self._target_dim] < x[index, self._target_dim]
                if np.any(mask):
                    new_x[index, self._target_dim] = self._reversed_position_set[self._target_dim][np.argmax(mask)]
                # boundary hit : cannot go lower than the lowest value --> go up instead
                else:
                    new_x[index, self._target_dim] = self._position_set[self._target_dim][1]

        # next time, update the position for the next dimension of the vector
        self._target_dim += 1
        if self._target_dim >= x.shape[1]:
            self._target_dim = 0

        return new_x


# functions
def parse_moves(moves: ParallelMove | Sequence[ParallelMove] | Sequence[tuple[float, ParallelMove]],
                dtype: np.dtype) \
        -> tuple[list[float], list[ParallelMove]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    Args:
        moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
            sequence of tuples with format (probability: float, Move).
        dtype: a data type to set to the moves.

    Returns:
        The list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.abc.Sequence) or isinstance(moves, str):
        if isinstance(moves, ParallelMove):
            return [1.0], [moves]

        raise ValueError(f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a "
                         f"'ParallelMove', a sequence of 'ParallelMove's or a sequence of tuples "
                         f"'(probability: float, 'ParallelMove')'.")

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
            raise ValueError(f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                             f"defining a move, expected a 'ParallelMove' or tuple "
                             f"'(probability: float, 'ParallelMove')'.")

    for move in parsed_moves:
        move.set_dtype(dtype)

    if sum(parsed_probabilities) != 1:
        _sum = sum(parsed_probabilities)
        parsed_probabilities = [proba / _sum for proba in parsed_probabilities]

    return parsed_probabilities, parsed_moves
