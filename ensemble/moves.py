# coding: utf-8
# Created on 29/07/2021 08:58
# Author : matteo

# ====================================================
# imports
import numpy as np
from abc import ABC, abstractmethod

from typing import List, Optional, Sequence, Tuple

from ..__moves import Move


# ====================================================
# code
class EnsembleMove(Move, ABC):

    @abstractmethod
    def get_proposal(self, x: np.ndarray, c: List[np.ndarray]) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)
        :param c: set of complementary vectors x_[k] of shape (nb_walkers, ndim)

        :return: new proposed vector x of shape (ndim,)
        """
        pass


class Stretch(EnsembleMove):
    """
    Stretch move as defined in 'Goodman, J., Weare, J., 2010, Comm. App. Math. and Comp. Sci., 5, 65'
    """

    def __init__(self, a: float = 2., bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
            while bigger values make samples more spread out with a peak getting closer to 0.
        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        super().__init__(bounds=bounds)
        self.__a = a

    def __sample_z(self) -> float:
        """
        Get a sample from the distribution of Z :
             |  1 / sqrt(z)     if z in [1/a, a]
             |  0               otherwise

        :return: a sample from Z.
        """
        return (np.random.rand() * self.__a + 2) ** 2 / (4 * self.__a)

    def get_proposal(self, x: np.ndarray, c: List[np.ndarray]) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)
        :param c: set of complementary vectors x_[k] of shape (nb_walkers, ndim)

        :return: new proposed vector x of shape (ndim,)
        """
        # pick X_j at random from the complementary set
        x_j = c[np.random.randint(0, len(c))]
        # sample z
        z = self.__sample_z()
        # move
        return x_j + z * (x - x_j)
