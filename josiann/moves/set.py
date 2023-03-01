# coding: utf-8
# Created on 13/01/2023 09:12
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from abc import ABC

import numpy.typing as npt
from typing import Sequence
from typing import TYPE_CHECKING

from josiann.moves.base import Move
from josiann.moves.base import State
from josiann.moves.ensemble import Stretch
from josiann.errors import ShapeError

if TYPE_CHECKING:
    import josiann.typing as jot


# ====================================================
# code
class SetMove(Move, ABC):
    """
    Base class for building moves that work on a discrete set of valid positions.
    """


class SetStep(SetMove):
    """
    Step within a fixed set of possible values for x. For each dimension, the position immediately before or after x
        will be chosen at random when stepping.

    Args:
        position_set: sets of only possible values for x in each dimension.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(
        self,
        position_set: Sequence[Sequence[float]],
        bounds: npt.NDArray[np.float64] | None = None,
    ):
        super().__init__(bounds=bounds)

        if not all(isinstance(p, (Sequence, np.ndarray)) for p in position_set):
            raise ShapeError(
                "'position_set' parameter should be an array of possible position values of shape "
                "(dimensions, nb_values) (nb_values can be different for each dimension)."
            )

        self._position_set = [np.sort(p) for p in position_set]
        self._reversed_position_set = [v[::-1] for v in self._position_set]
        self._target_dim = 0

    def _get_proposal(
        self, x: npt.NDArray[jot.DT_ARR], state: State
    ) -> npt.NDArray[jot.DT_ARR]:
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
            mask = self._position_set[self._target_dim] > x[self._target_dim]
            if np.any(mask):
                new_x[self._target_dim] = self._position_set[self._target_dim][
                    np.argmax(mask)
                ]
            else:
                new_x[self._target_dim] = x[self._target_dim]

        else:
            mask = self._reversed_position_set[self._target_dim] < x[self._target_dim]
            if np.any(mask):
                new_x[self._target_dim] = self._reversed_position_set[self._target_dim][
                    np.argmax(mask)
                ]
            else:
                new_x[self._target_dim] = x[self._target_dim]

        self._target_dim += 1
        if self._target_dim >= len(x):
            self._target_dim = 0

        return new_x


class SetStretch(SetMove, Stretch):
    """
    Fusion of the Set and Stretch moves. We exploit multiple walkers in parallel a move each to the closest point
        in the set of possible positions instead of the point proposed by the stretch.

    Args:
        position_set: sets of only possible values for x in each dimension.
        a: parameter for tuning the distribution of Z. Smaller values make samples tightly distributed around 1
            while bigger values make samples more spread out with a peak getting closer to 0.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(
        self,
        position_set: Sequence[Sequence[float]],
        a: float = 2.0,
        bounds: npt.NDArray[np.float64] | None = None,
    ):
        super().__init__(a=a, bounds=bounds)

        self._position_set = [np.sort(p) for p in position_set]

    def _find_nearest(self, vector: npt.NDArray[jot.DT_ARR]) -> npt.NDArray[jot.DT_ARR]:
        """
        Find the nearest values in <array> for each element in <vector>.

        Args:
            vector: an array of values for which to find the nearest values.

        Returns:
            An array with the nearest values from <vector> in <array>.
        """
        for index, value in enumerate(vector):
            vector[index] = self._position_set[index][
                np.nanargmin(np.abs(self._position_set[index] - value))
            ]

        return vector

    def _get_proposal(
        self, x: npt.NDArray[jot.DT_ARR], state: State
    ) -> npt.NDArray[jot.DT_ARR]:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        # pick X_j at random from the complementary set
        x_j = state.complementary_set[
            np.random.randint(0, len(state.complementary_set))
        ]
        # sample z
        r = state.iteration / state.max_iter
        a = (1.5 - self._a) * r + self._a
        z = self._sample_z(a)

        proposal = x_j + z * (x - x_j)

        # move
        return self._find_nearest(proposal)
