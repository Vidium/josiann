# coding: utf-8
# Created on 29/07/2021 09:30
# Author : matteo

# ====================================================
# imports
import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, Sequence, Tuple, List, Union


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

    def set_bounds(self, bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]]) -> None:
        """
        Set bounds for the Move.

        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension or a single
            (min, max) tuple of bounds to set for all dimensions.
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


# Moves independent from other walkers
class SingleMove(Move, ABC):

    @abstractmethod
    def get_proposal(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
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

    def get_proposal(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
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

    def get_proposal(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)

        :return: new proposed vector x of shape (ndim,)
        """
        return self._valid_proposal(np.random.multivariate_normal(x, self.__cov))


# Moves depending on other walkers
class EnsembleMove(Move, ABC):

    @abstractmethod
    def get_proposal(self,
                     x: np.ndarray,
                     c: List[np.ndarray],
                     iteration: int,
                     max_iter: int,
                     *args, **kwargs) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)
        :param c: set of complementary vectors x_[k] of shape (nb_walkers, ndim)
        :param iteration: current iteration number.
        :param max_iter: maximum iteration number.

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
        self._a = a

    @staticmethod
    def _sample_z(a: float) -> float:
        """
        Get a sample from the distribution of Z :
             |  1 / sqrt(z)     if z in [1/a, a]
             |  0               otherwise

        :param a: parameter for tuning the distribution of Z.

        :return: a sample from Z.
        """
        return (np.random.rand() * a + 2) ** 2 / (4 * a)

    def get_proposal(self,
                     x: np.ndarray,
                     c: List[np.ndarray],
                     iteration: int,
                     max_iter: int,
                     *args, **kwargs) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)
        :param c: set of complementary vectors x_[k] of shape (nb_walkers-1, ndim)
        :param iteration: current iteration number.
        :param max_iter: maximum iteration number.

        :return: new proposed vector x of shape (ndim,)
        """
        # pick X_j at random from the complementary set
        x_j = c[np.random.randint(0, len(c))]
        # sample z
        z = self._sample_z(self._a)
        # move
        return self._valid_proposal(x_j + z * (x - x_j))


class StretchAdaptative(Stretch):
    """
    Stretch move as defined in 'Goodman, J., Weare, J., 2010, Comm. App. Math. and Comp. Sci., 5, 65' with decreasing
    'a' parameter.
    """

    def __init__(self, a: float = 2., bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
            while bigger values make samples more spread out with a peak getting closer to 0.
        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        super().__init__(a=a, bounds=bounds)

    def get_proposal(self,
                     x: np.ndarray,
                     c: List[np.ndarray],
                     iteration: int,
                     max_iter: int,
                     *args, **kwargs) -> np.ndarray:
        """
        Generate a new proposed vector x.

        :param x: current vector x of shape (ndim,)
        :param c: set of complementary vectors x_[k] of shape (nb_walkers-1, ndim)
        :param iteration: current iteration number.
        :param max_iter: maximum iteration number.

        :return: new proposed vector x of shape (ndim,)
        """
        # pick X_j at random from the complementary set
        x_j = c[np.random.randint(0, len(c))]
        # sample z
        r = iteration / max_iter
        a = (1.5-self._a)*r+self._a
        z = self._sample_z(a)
        # move
        return self._valid_proposal(x_j + z * (x - x_j))
