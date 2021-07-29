# coding: utf-8
# Created on 29/07/2021 09:30
# Author : matteo

# ====================================================
# imports
import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, Sequence, Tuple

from ..__moves import Move


# ====================================================
# code
class SingleMove(Move, ABC):

    @abstractmethod
    def get_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)

        :return: new proposed vector x of shape (ndim,)
        """
        pass


class RandomStep(SingleMove):
    """
    Simple random step within a radius of (-0.5 * magnitude) to (+0.5 * magnitude) around x.
    """

    def __init__(self, magnitude: float, bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param magnitude: size of the random step is (-0.5 * magnitude) to (+0.5 * magnitude)
        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        super().__init__(bounds=bounds)
        self.__magnitude = magnitude

    def get_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)

        :return: new proposed vector x of shape (ndim,)
        """
        return self._valid_proposal(x + self.__magnitude * (np.random.random(len(x)) - 0.5))


class Metropolis(SingleMove):
    """
    Metropolis step obtained from a multivariate normal distribution with mean <x> and covariance matrix <variances>
    """

    def __init__(self, variances: np.ndarray, bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param variances: list of variances between dimensions, which will be set as the diagonal of the covariance
            matrix.
        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        super().__init__(bounds=bounds)
        self.__cov = np.diag(variances)

    def get_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)

        :return: new proposed vector x of shape (ndim,)
        """
        return self._valid_proposal(np.random.multivariate_normal(x, self.__cov))
