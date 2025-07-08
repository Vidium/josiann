# Created on 05/08/2021 11:22
# Author : matteo

"""
Backup class for storing previously computed costs at visited position vectors.
"""

# ====================================================
# imports
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

# ====================================================
# code
EVALUATION = tuple[int, float]
EV = TypeVar("EV", EVALUATION, list[EVALUATION])


class Backup(ABC, Generic[EV]):
    """
    Object for storing previously computed function evaluations at given position vectors. This is only available
        when using SetStep moves since they offer a decent probability of hitting the exact same position vector
        twice whereas this probability is ~0 for over moves.

    Args:
        active: set this backup object to active ? (don't store anything if inactive.)
    """

    # region magic methods
    def __init__(self, active: bool):
        self.active = active

    def __repr__(self) -> str:
        return f"Backup: {'active' if self.active else 'no'}"

    # endregion

    # region methods
    @abstractmethod
    def save(self, position: npt.NDArray[np.float64 | np.int64], evaluation: EV) -> None:
        """
        Store computed function evaluations at given position vector.

        Args:
            position: position vector of the function evaluations.
            evaluation: function evaluation: number of evaluations, mean of evaluations.
        """

    @abstractmethod
    def get_previous_evaluations(self, position: npt.NDArray[np.float64 | np.int64]) -> EV:
        """
        Get stored last function evaluations at given position vector.

        Args:
            position: position vector of the function evaluations.

        Returns:
            Stored function evaluation: number of evaluations, mean of evaluations. (defaults to (0, 0)).
        """

    # endregion


class SequentialBackup(Backup[EVALUATION]):
    # region magic methods
    def __init__(self, active: bool = False):
        super().__init__(active)

        self._backup_array: dict[tuple[np.float64 | np.int64, ...], EVALUATION] = {}

    # endregion

    # region methods
    def save(self, position: npt.NDArray[np.float64 | np.int64], evaluation: EVALUATION) -> None:
        """
        Store computed function evaluations at given position vector.

        Args:
            position: position vector of the function evaluations.
            evaluation: function evaluation: number of evaluations, mean of evaluations.
        """
        position_tuple = tuple(position)
        if self.active:
            self._backup_array[position_tuple] = evaluation

    def get_previous_evaluations(self, position: npt.NDArray[np.float64 | np.int64]) -> EVALUATION:
        """
        Get stored last function evaluations at given position vector.

        Args:
            position: position vector of the function evaluations.

        Returns:
            Stored function evaluation: number of evaluations, mean of evaluations. (defaults to (0, 0)).
        """
        return self._backup_array.get(tuple(position), (0, 0.0))

    # endregion
