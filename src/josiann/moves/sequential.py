"""Simple sequential move functions."""

# ====================================================
# imports
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.moves.base import Move, State


# ====================================================
# code
class RandomStep(Move):
    """
    Simple random step within a radius of (-0.5 * magnitude) to (+0.5 * magnitude) around x.
    """

    # region magic methods
    def __init__(
        self,
        *,
        magnitude: float,
        bounds: npt.NDArray[jot.DType] | None = None,
        repr_attributes: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        """
        Instantiate a Move.

        Args:
            magnitude: size of the random step is (-0.5 * magnitude) to (+0.5 * magnitude)
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
            repr_attributes: list of attribute names to include in the string representation of this Move.
        """
        super().__init__(bounds=bounds, repr_attributes=("_magnitude",) + repr_attributes, **kwargs)

        self._magnitude = magnitude

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
        target_dim = np.random.randint(len(x))
        increment = np.zeros(len(x), dtype=x.dtype)
        increment[target_dim] = self._magnitude * (np.random.random() - 0.5)

        return x + increment  # type: ignore[return-value]

    # endregion


class Metropolis(Move):
    """
    Metropolis step obtained from a multivariate normal distribution with mean <x> and covariance matrix <variances>
    """

    # region magic methods
    def __init__(
        self,
        *,
        variances: npt.NDArray[np.float64],
        bounds: npt.NDArray[jot.DType] | None = None,
        repr_attributes: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        """
        Instantiate a Move.

        Args:
            variances: list of variances between dimensions, which will be set as the diagonal of the covariance matrix.
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
            repr_attributes: list of attribute names to include in the string representation of this Move.
        """
        super().__init__(bounds=bounds, repr_attributes=("_cov",) + repr_attributes, **kwargs)

        self._cov = np.diag(variances)

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
        return np.random.multivariate_normal(x, self._cov)  # type: ignore[return-value]

    # endregion


class Metropolis1D(Move):
    """
    Metropolis step obtained from a uni-variate normal distribution with mean <x> and variance <variance>
    """

    # region magic methods
    def __init__(
        self,
        *,
        variance: float,
        bounds: npt.NDArray[jot.DType] | None = None,
        repr_attributes: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        """
        Instantiate a Move.

        Args:
            variance: the variance for the normal distribution.
            bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
            repr_attributes: list of attribute names to include in the string representation of this Move.
        """
        super().__init__(bounds=bounds, repr_attributes=("_var",) + repr_attributes, **kwargs)

        self._var = float(variance)

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
        target_dim = np.random.randint(len(x))
        x[target_dim] = np.random.normal(x[target_dim], self._var)

        return x

    # endregion
