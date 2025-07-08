# Created on 13/01/2023 09:16
# Author : matteo

"""Move functions that use multiple walkers to better solve some types of problems."""

# ====================================================
# imports
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.moves.base import Move, State


# ====================================================
# code
class EnsembleMove(Move, ABC):
    """
    Base class for building moves that require an ensemble of walkers to evolve in parallel.
    """


class Stretch(EnsembleMove):
    """
    Stretch move as defined in 'Goodman, J., Weare, J., 2010, Comm. App. Math. and Comp. Sci., 5, 65'
    """

    # region magic methods
    def __init__(
        self,
        *,
        a: float = 2.0,
        bounds: npt.NDArray[jot.DType] | None = None,
        repr_attributes: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        """
        Instantiate a Move.

        Args:
            position_set: sets of only possible values for x in each dimension.
            a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
                while bigger values make samples more spread out with a peak getting closer to 0.
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
            repr_attributes: tuple of attribute names to include in the move's representation.
        """
        super().__init__(bounds=bounds, repr_attributes=("_a",) + repr_attributes, **kwargs)

        self._a = a

    # endregion

    # region methods
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

    def _get_proposal(self, x: npt.NDArray[jot.DT_ARR], state: State) -> npt.NDArray[jot.DT_ARR]:
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
        return x_j + z * (x - x_j)  # type: ignore[no-any-return]

    # endregion


class StretchAdaptive(Stretch):
    """
    Stretch move as defined in 'Goodman, J., Weare, J., 2010, Comm. App. Math. and Comp. Sci., 5, 65' with decreasing
    'a' parameter.
    """

    # region methods
    def _get_proposal(self, x: npt.NDArray[jot.DT_ARR], state: State) -> npt.NDArray[jot.DT_ARR]:
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
        # move
        return x_j + z * (x - x_j)  # type: ignore[no-any-return]

    # endregion
