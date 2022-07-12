# coding: utf-8
# Created on 16/06/2022 11:30
# Author : matteo

# ====================================================
# imports
import numpy as np


# ====================================================
# code
class Backup:
    """
    Object for storing previously computed function evaluations at given position vectors. This is only available
        when using SetStep moves since they offer a decent probability of hitting the exact same position vector
        twice whereas this probability is ~0 for over moves.
    """
    __slots__ = 'active', '_backup_array'

    def __init__(self,
                 nb_parallel_problems: int,
                 active: bool = False):
        """
        Args:
            nb_parallel_problems: number of parallel problems.
            active: set this backup object to active ? (don't store anything if inactive.)
        """
        self.active = active
        self._backup_array: list[dict[tuple, tuple[int, float]]] = [{} for _ in range(nb_parallel_problems)]

    def save(self,
             positions: np.ndarray,
             evaluation: list[tuple[int, float]]) -> None:
        """
        Store computed function evaluations at given position vector.

        Args:
            positions: position vectors of the function evaluations.
            evaluation: function evaluations: (number of evaluations, mean of evaluations).
        """
        for problem_index, position in enumerate(positions):
            position_tuple = tuple(position)
            if self.active:
                self._backup_array[problem_index][position_tuple] = evaluation[problem_index]

    def get_previous_evaluations(self,
                                 positions: np.ndarray) -> list[tuple[int, float]]:
        """
        Get stored last function evaluations at given position vector.

        Args:
            positions: position vectors of the function evaluations.

        Returns:
            Stored function evaluation: number of evaluations, mean of evaluations. (defaults to (0, 0)).
        """
        return [self._backup_array[problem_index].get(tuple(position), (0, 0.))
                for problem_index, position in enumerate(positions)]
