# coding: utf-8
# Created on 05/08/2021 11:22
# Author : matteo

"""
Backup class for storing previously computed costs at visited position vectors.
"""

# ====================================================
# imports
import numpy as np
from multiprocessing.managers import BaseManager, NamespaceProxy  # type: ignore


# ====================================================
# code
class Backup:
    """
    Object for storing previously computed function evaluations at given position vectors. This is only available
        when using SetStep moves since they offer a decent probability of hitting the exact same position vector
        twice whereas this probability is ~0 for over moves.

    :param active: set this backup object to active ? (don't store anything if inactive.)
    """

    def __init__(self, active: bool = False):
        self.active = active
        self.__backup_array: dict[tuple, tuple[int, float]] = {}

    def save(self, position: np.ndarray, evaluation: tuple[int, float]) -> None:
        """
        Store computed function evaluations at given position vector.

        :param position: position vector of the function evaluations.
        :param evaluation: function evaluation: number of evaluations, mean of evaluations.
        """
        position_tuple = tuple(position)
        if self.active:
            self.__backup_array[position_tuple] = evaluation

    def get_previous_evaluations(self, position: np.ndarray) -> tuple[int, float]:
        """
        Get stored last function evaluations at given position vector.

        :param position: position vector of the function evaluations.

        :return: stored function evaluation: number of evaluations, mean of evaluations. (defaults to (0, 0))
        """
        return self.__backup_array.get(tuple(position), (0, 0.))


class BackupManager(BaseManager):
    """
    Manager for passing the Backup objects during multiprocessing.
    """


class BackupProxy(NamespaceProxy):
    """
    Proxy for accessing methods of the Backup objects during multiprocessing.
    """
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'save', 'get_previous_evaluations')

    def save(self, position: np.ndarray, evaluation: tuple[int, float]) -> None:
        """
        Proxy for Backup.save() method.
        """
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.save.__name__, (position, evaluation))

    def get_previous_evaluations(self, position: np.ndarray) -> tuple[int, float]:
        """
        Proxy for Backup.get_previous_evaluations() method.
        """
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.get_previous_evaluations.__name__, (position,))


BackupManager.register('Backup', Backup, BackupProxy)
