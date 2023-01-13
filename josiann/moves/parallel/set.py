# coding: utf-8
# Created on 13/01/2023 09:25
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Sequence

from josiann.moves.base import State
from josiann.moves.set import SetMove
from josiann.moves.parallel.base import ParallelMove
from josiann.errors import ShapeError

import josiann.typing as jot


# ====================================================
# code
class ParallelSetStep(ParallelMove, SetMove):
    """
    Step within a fixed set of possible values for x. For each dimension, the position immediately before or after x
        will be chosen at random when stepping.

    Args:
        position_set: sets of only possible values for x in each dimension.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(
        self,
        position_set: Sequence[npt.NDArray[jot.DT_ARR]] | npt.NDArray[jot.DT_ARR],
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

    def __repr__(self) -> str:
        return f"[Move] ParallelSetStep : {self._position_set}"

    def _get_proposal(
        self, x: npt.NDArray[jot.DT_ARR], state: State | None
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

        # loop on vectors in x
        for index in range(len(x)):
            # for each, draw at random if the position increases ...
            if np.random.rand() > 0.5:
                mask = self._position_set[self._target_dim] > x[index, self._target_dim]
                if np.any(mask):
                    new_x[index, self._target_dim] = self._position_set[
                        self._target_dim
                    ][np.argmax(mask)]
                # boundary hit : cannot go higher than the highest value --> go down instead
                else:
                    new_x[index, self._target_dim] = self._position_set[
                        self._target_dim
                    ][-2]

            # ... or decreases
            else:
                mask = (
                    self._reversed_position_set[self._target_dim]
                    < x[index, self._target_dim]
                )
                if np.any(mask):
                    new_x[index, self._target_dim] = self._reversed_position_set[
                        self._target_dim
                    ][np.argmax(mask)]
                # boundary hit : cannot go lower than the lowest value --> go up instead
                else:
                    new_x[index, self._target_dim] = self._position_set[
                        self._target_dim
                    ][1]

        # next time, update the position for the next dimension of the vector
        self._target_dim += 1
        if self._target_dim >= x.shape[1]:
            self._target_dim = 0

        return new_x
