# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from abc import ABC
from abc import abstractmethod
from attrs import frozen

import numpy.typing as npt
from typing import Sequence
from typing import TYPE_CHECKING

from josiann.errors import ShapeError

if TYPE_CHECKING:
    import josiann.typing as jot


# ====================================================
# code
@frozen
class State:
    """
    Object for describing the current state of the SA algorithm.

    complementary_set: set of complementary vectors x_[k] of shape (nb_walkers-1, ndim)
    iteration: current iteration number.
    max_iter: maximum iteration number.
    """

    complementary_set: npt.NDArray[jot.DType]
    iteration: int
    max_iter: int


class Move(ABC):
    """
    Base abstract class for defining how positions evolve in the SA algorithm.

    Args:
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self, bounds: npt.NDArray[jot.DType] | None = None):
        self._bounds = bounds

    def set_bounds(
        self, bounds: tuple[float, float] | Sequence[tuple[float, float]] | None
    ) -> None:
        if bounds is not None:
            bounds_arr = np.array(bounds)

            if bounds_arr.ndim in (1, 2):
                self._bounds = bounds_arr

            else:
                raise ShapeError(f"Invalid bounds with shape {bounds_arr.shape}.")

    @abstractmethod
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
        pass

    def get_proposal(
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
        return self._valid_proposal(self._get_proposal(x, state).astype(x.dtype))

    def _valid_proposal(self, x: npt.NDArray[jot.DT_ARR]) -> npt.NDArray[jot.DT_ARR]:
        """
        Get valid proposal within defined bounds.

        Args:
            a 'raw' proposal.

        Returns
            A proposal with values restricted with the defined bounds.
        """
        if self._bounds is not None:
            return np.minimum(np.maximum(x, self._bounds[:, 0]), self._bounds[:, 1])

        return x
