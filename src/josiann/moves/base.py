# ====================================================
# imports
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from attrs import frozen

import josiann.typing as jot
from josiann.errors import ShapeError


# ====================================================
# code
@frozen
class State:
    """
    Object for describing the current state of the SA algorithm.

    Args:
        complementary_set: matrix of position vectors from walkers other than the one to update.
        iteration: current iteration number.
        max_iter: maximum iteration number.
    """

    complementary_set: npt.NDArray[jot.DType]  #: matrix of position vectors from walkers other than the one to update.
    iteration: int  #: current iteration number.
    max_iter: int  #: maximum iteration number.


class Move(ABC):
    """
    Base abstract class for defining how positions evolve in the SA algorithm.
    """

    # region magic methods
    def __init__(
        self,
        *,
        bounds: npt.NDArray[jot.DType] | None = None,
        repr_attributes: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        """
        Instantiate a Move.

        Args:
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
            repr_attributes: list of attribute names to include in the string representation of this Move.
        """
        self._bounds = bounds
        self._repr_attributes = tuple(repr_attributes)

    def __repr__(self) -> str:
        with np.printoptions(precision=4):
            repr_str = (
                f"[Move] {type(self).__name__}("
                f"{', '.join([str(getattr(self, attr_name)) for attr_name in self._repr_attributes])}"
                f")"
            )

        return repr_str

    # endregion

    # region methods
    def set_bounds(self, bounds: tuple[float, float] | Sequence[tuple[float, float]] | None) -> None:
        """
        Set bounds for the move.

        Args:
            bounds: sequence of (min, max) bounds for values to propose in each dimension.
        """
        if bounds is not None:
            bounds_arr = np.array(bounds)

            if bounds_arr.ndim in (1, 2):
                self._bounds = bounds_arr

            else:
                raise ShapeError(f"Invalid bounds with shape {bounds_arr.shape}.")

    @abstractmethod
    def _get_proposal(self, x: npt.NDArray[jot.DT_ARR], state: State) -> npt.NDArray[jot.DT_ARR]:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        pass

    def get_proposal(self, x: npt.NDArray[jot.DT_ARR], state: State) -> npt.NDArray[jot.DT_ARR]:
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
            x: a 'raw' proposal.

        Returns
            A proposal with values restricted with the defined bounds.
        """
        if self._bounds is not None:
            return np.minimum(np.maximum(x, self._bounds[:, 0]), self._bounds[:, 1])  # type: ignore[no-any-return]

        return x

    # endregion
