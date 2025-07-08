# Created on 16/06/2022 11:30
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from josiann.backup.backup import EVALUATION, Backup


# ====================================================
# code
class ParallelBackup(Backup[list[EVALUATION]]):
    """
    Object for storing previously computed function evaluations at given position vectors. This is only available
        when using SetStep moves since they offer a decent probability of hitting the exact same position vector
        twice whereas this probability is ~0 for over moves.
    """

    # region magic methods
    def __init__(self, active: bool, nb_parallel_problems: int):
        """
        Args:
            active: set this backup object to active ? (don't store anything if inactive.)
            nb_parallel_problems: number of parallel problems.
        """
        super().__init__(active)

        self._backup_array: list[dict[tuple[np.float64 | np.int64, ...], tuple[int, float]]] = [
            {} for _ in range(nb_parallel_problems)
        ]

    # endregion

    # region methods
    def save(
        self,
        positions: npt.NDArray[np.float64 | np.int64],
        evaluation: list[EVALUATION],
    ) -> None:
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

    def get_previous_evaluations(self, positions: npt.NDArray[np.float64 | np.int64]) -> list[EVALUATION]:
        """
        Get stored last function evaluations at given position vector.

        Args:
            positions: position vectors of the function evaluations.

        Returns:
            Stored function evaluation: number of evaluations, mean of evaluations. (defaults to (0, 0)).
        """
        return [
            self._backup_array[problem_index].get(tuple(position), (0, 0.0))
            for problem_index, position in enumerate(positions)
        ]

    # endregion
