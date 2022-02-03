# coding: utf-8
# Created on 29/07/2021 09:30
# Author : matteo

"""
Defines Moves for updating the position vectors.
Moves can be walker-independent, where each walker is updated without information about other walkers,
or walker-dependent.
"""

# ====================================================
# imports
import collections.abc
import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, Sequence, Union

from .utils import State
from .name_utils import ShapeError


# ====================================================
# code
class Move(ABC):
    """
    Base abstract class for defining how positions evolve in the SA algorithm.

    Args:
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        self.__bounds = np.array(bounds) if bounds is not None else None

    def set_bounds(self,
                   bounds: Optional[Union[tuple[float, float], Sequence[tuple[float, float]]]]) -> None:
        """
        Set bounds for the Move.

        Args:
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension or a single
                (min, max) tuple of bounds to set for all dimensions.
        """
        self.__bounds = np.array(bounds) if bounds is not None else None

    @abstractmethod
    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """

    def _valid_proposal(self,
                        x: np.ndarray) -> np.ndarray:
        """
        Get valid proposal within defined bounds.

        Args:
            a 'raw' proposal.

        Returns
            A proposal with values restricted with the defined bounds.
        """
        if self.__bounds is not None:
            return np.minimum(np.maximum(x, self.__bounds[:, 0]), self.__bounds[:, 1])

        return x


# Moves independent of other walkers
class RandomStep(Move):
    """
    Simple random step within a radius of (-0.5 * magnitude) to (+0.5 * magnitude) around x.

    Args:
        magnitude: size of the random step is (-0.5 * magnitude) to (+0.5 * magnitude)
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 magnitude: float,
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(bounds=bounds)
        self.__magnitude = magnitude

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        target_dim = np.random.randint(len(x))
        increment = np.zeros(len(x))
        increment[target_dim] = self.__magnitude * (np.random.random() - 0.5)

        return self._valid_proposal(x + increment)


class SetStep(Move):
    """
    Step within a fixed set of possible values for x. For each dimension, the position immediately before or after x
        will be chosen at random when stepping.

    Args:
        position_set: sets of only possible values for x in each dimension.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 position_set: Sequence[Sequence[float]],
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(bounds=bounds)

        if not all(isinstance(p, (Sequence, np.ndarray)) for p in position_set):
            raise ShapeError("'position_set' parameter should be an array of possible position values of shape "
                             "(dimensions, nb_values) (nb_values can be different for each dimension).")

        self.__position_set = [np.sort(p) for p in position_set]
        self.__reversed_position_set = [v[::-1] for v in self.__position_set]
        self.__target_dim = 0

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        new_x = x.copy()

        if np.random.rand() > 0.5:
            mask = self.__position_set[self.__target_dim] > x[self.__target_dim]
            if np.any(mask):
                new_x[self.__target_dim] = self.__position_set[self.__target_dim][np.argmax(mask)]
            else:
                new_x[self.__target_dim] = x[self.__target_dim]

        else:
            mask = self.__reversed_position_set[self.__target_dim] < x[self.__target_dim]
            if np.any(mask):
                new_x[self.__target_dim] = self.__reversed_position_set[self.__target_dim][np.argmax(mask)]
            else:
                new_x[self.__target_dim] = x[self.__target_dim]

        self.__target_dim += 1
        if self.__target_dim >= len(x):
            self.__target_dim = 0

        return self._valid_proposal(new_x)


class Metropolis(Move):
    """
    Metropolis step obtained from a multivariate normal distribution with mean <x> and covariance matrix <variances>

    Args:
        variances: list of variances between dimensions, which will be set as the diagonal of the covariance
            matrix.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 variances: np.ndarray, bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(bounds=bounds)
        self.__cov = np.diag(variances)

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        return self._valid_proposal(np.random.multivariate_normal(x, self.__cov))


class Metropolis1D(Move):
    """
    Metropolis step obtained from a uni-variate normal distribution with mean <x> and variance <variance>

    Args:
        variance: the variance.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 variance: float,
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(bounds=bounds)
        self.__var = float(variance)

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        target_dim = np.random.randint(len(x))
        x[target_dim] = np.random.normal(x[target_dim], self.__var)

        proposal = self._valid_proposal(x)

        return proposal


# Moves depending on other walkers
class EnsembleMove(Move, ABC):
    """
    Base class for building moves that require an ensemble of walkers to evolve in parallel.
    """


class Stretch(EnsembleMove):
    """
    Stretch move as defined in 'Goodman, J., Weare, J., 2010, Comm. App. Math. and Comp. Sci., 5, 65'

    Args:
        a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
            while bigger values make samples more spread out with a peak getting closer to 0.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 a: float = 2.,
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(bounds=bounds)
        self._a = a

    @staticmethod
    def _sample_z(a: float) -> float:
        """
        Get a sample from the distribution of Z :
             |  1 / sqrt(z)     if z in [1/a, a]
             |  0               otherwise

        Args:
            a: parameter for tuning the distribution of Z.

        Returns:
            A sample from Z.
        """
        return (np.random.rand() * a + 2) ** 2 / (4 * a)

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        # pick X_j at random from the complementary set
        x_j = state.complementary_set[np.random.randint(0, len(state.complementary_set))]
        # sample z
        z = self._sample_z(self._a)
        # move
        return self._valid_proposal(x_j + z * (x - x_j))


class StretchAdaptive(Stretch):
    """
    Stretch move as defined in 'Goodman, J., Weare, J., 2010, Comm. App. Math. and Comp. Sci., 5, 65' with decreasing
    'a' parameter.

    Args:
        a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
            while bigger values make samples more spread out with a peak getting closer to 0.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 a: float = 2.,
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(a=a, bounds=bounds)

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        # pick X_j at random from the complementary set
        x_j = state.complementary_set[np.random.randint(0, len(state.complementary_set))]
        # sample z
        r = state.iteration / state.max_iter
        a = (1.5-self._a)*r+self._a
        z = self._sample_z(a)
        # move
        return self._valid_proposal(x_j + z * (x - x_j))


class SetStretch(Stretch):
    """
    Fusion of the Set and Stretch moves. We exploit multiple walkers in parallel a move each to the closest point
        in the set of possible positions instead of the point proposed by the stretch.

    Args:
        position_set: sets of only possible values for x in each dimension.
        a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
            while bigger values make samples more spread out with a peak getting closer to 0.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self,
                 position_set: Sequence[Sequence[float]],
                 a: float = 2.,
                 bounds: Optional[Sequence[tuple[float, float]]] = None):
        super().__init__(a=a, bounds=bounds)

        self.__position_set = [np.sort(p) for p in position_set]

    def _find_nearest(self,
                      vector: np.ndarray):
        """
        Find the nearest values in <array> for each element in <vector>.

        Args:
            vector: an array of values for which to find the nearest values.

        Returns:
            An array with the nearest values from <vector> in <array>.
        """
        for index, value in enumerate(vector):
            vector[index] = self.__position_set[index][np.nanargmin(np.abs(self.__position_set[index] - value))]

        return vector

    def get_proposal(self,
                     x: np.ndarray,
                     state: State) -> np.ndarray:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        # pick X_j at random from the complementary set
        x_j = state.complementary_set[np.random.randint(0, len(state.complementary_set))]
        # sample z
        r = state.iteration / state.max_iter
        a = (1.5 - self._a) * r + self._a
        z = self._sample_z(a)

        proposal = x_j + z * (x - x_j)

        # move
        return self._valid_proposal(self._find_nearest(proposal))


# functions
def parse_moves(moves: Union[Move, Sequence[Move], Sequence[tuple[float, Move]]],
                nb_walkers: int) -> tuple[list[float], list[Move]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    Args:
        moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
            sequence of tuples with format (probability: float, Move).
        nb_walkers: the number of parallel walkers in the ensemble.

    Returns:
        The list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.abc.Sequence) or isinstance(moves, str):
        if isinstance(moves, Move):
            if issubclass(type(moves), EnsembleMove) and nb_walkers < 2:
                raise ValueError('Ensemble moves require at least 2 walkers to be used.')

            return [1.0], [moves]

        raise ValueError(f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a "
                         f"'Move', a sequence of 'Move's or a sequence of tuples "
                         f"'(probability: float, 'Move')'.")

    parsed_probabilities = []
    parsed_moves = []

    for move in moves:
        if isinstance(move, Move):
            if issubclass(type(move), EnsembleMove) and nb_walkers < 2:
                raise ValueError('Ensemble moves require at least 2 walkers to be used.')

            parsed_probabilities.append(1.0)
            parsed_moves.append(move)

        elif isinstance(move, tuple):
            if len(move) == 2 and isinstance(move[0], float) and isinstance(move[1], Move):
                if issubclass(type(move[1]), EnsembleMove) and nb_walkers < 2:
                    raise ValueError('Ensemble moves require at least 2 walkers to be used.')

                parsed_probabilities.append(move[0])
                parsed_moves.append(move[1])

            else:
                raise ValueError(f"Invalid format for tuple '{move}', expected '(probability: float, Move)'.")

        else:
            raise ValueError(f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                             f"defining a move, expected a 'Move' or tuple '(probability: float, 'Move')'.")

    if sum(parsed_probabilities) != 1:
        _sum = sum(parsed_probabilities)
        parsed_probabilities = [proba / _sum for proba in parsed_probabilities]

    return parsed_probabilities, parsed_moves
