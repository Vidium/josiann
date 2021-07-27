# coding: utf-8
# Created on 23/07/2021 23:28
# Author : matteo

# ====================================================
# imports
import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, Sequence, Tuple


# ====================================================
# code
class Move(ABC):
    """
    Base abstract class for
    """

    def __init__(self, bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        self.__bounds = np.array(bounds) if bounds is not None else None

    @abstractmethod
    def get_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)

        :return: new proposed vector x of shape (ndim,)
        """
        pass

    def set_bounds(self, bounds: Optional[Sequence[Tuple[float, float]]]) -> None:
        """
        Set bounds for the Move.

        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        self.__bounds = np.array(bounds) if bounds is not None else None

    def _valid_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Get valid proposal within defined bounds.

        :param x: a 'raw' proposal.
        :return: a proposal with values restricted with the defined bounds.
        """
        if self.__bounds is not None:
            return np.minimum(np.maximum(x, self.__bounds[:, 0]), self.__bounds[:, 1])

        return x


class RandomStep(Move):
    """
    Simple random step within a radius of (-0.5 * magnitude) to (+0.5 * magnitude) around x.
    """

    def __init__(self, magnitude: float, bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param magnitude: size of the random step is (-0.5 * magnitude) to (+0.5 * magnitude)
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


class Metropolis(Move):
    """

    """

    def __init__(self, variances: np.ndarray, bounds: Optional[Sequence[Tuple[float, float]]] = None):
        super().__init__(bounds=bounds)
        self.__cov = np.diag(variances)

    def get_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)

        :return: new proposed vector x of shape (ndim,)
        """
        return self._valid_proposal(np.random.multivariate_normal(x, self.__cov))
