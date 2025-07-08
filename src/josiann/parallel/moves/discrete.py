"""Discrete move functions designed to work with josiann's parallel mode."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.moves.base import State
from josiann.moves.discrete import DiscreteMove
from josiann.parallel.moves.base import ParallelMove


class ParallelSetStep(ParallelMove, DiscreteMove):
    """
    Step within a fixed set of possible values for x. For each dimension, the position immediately before or after x
    will be chosen at random when stepping.
    """

    # region magic methods
    def __init__(
        self,
        *,
        position_set: Sequence[Sequence[float] | npt.NDArray[np.floating[Any]]],
        bounds: npt.NDArray[jot.DType] | None = None,
        repr_attributes: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        """
        Instantiate a Move.

        Args:
            position_set: sets of only possible values for x in each dimension.
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
            repr_attributes: tuple of attribute names to include in the move's representation.
        """
        super().__init__(
            position_set=position_set,
            bounds=bounds,
            repr_attributes=repr_attributes,
            **kwargs,
        )

        self._reversed_position_set = [v[::-1] for v in self._position_set]
        self._target_dim = 0

    # endregion

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
        new_x = x.copy()

        # loop on vectors in x
        for index in range(len(x)):
            # for each, draw at random if the position increases ...
            if np.random.rand() > 0.5:
                mask = self._position_set[self._target_dim] > x[index, self._target_dim]
                if np.any(mask):
                    new_x[index, self._target_dim] = self._position_set[self._target_dim][np.argmax(mask)]
                # boundary hit : cannot go higher than the highest value --> go down instead
                else:
                    new_x[index, self._target_dim] = self._position_set[self._target_dim][-2]

            # ... or decreases
            else:
                mask = self._reversed_position_set[self._target_dim] < x[index, self._target_dim]
                if np.any(mask):
                    new_x[index, self._target_dim] = self._reversed_position_set[self._target_dim][np.argmax(mask)]
                # boundary hit : cannot go lower than the lowest value --> go up instead
                else:
                    new_x[index, self._target_dim] = self._position_set[self._target_dim][1]

        # next time, update the position for the next dimension of the vector
        self._target_dim += 1
        if self._target_dim >= x.shape[1]:
            self._target_dim = 0

        return new_x

    # endregion
