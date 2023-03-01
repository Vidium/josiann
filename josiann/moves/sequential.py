# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import TYPE_CHECKING

from josiann.moves.base import Move
from josiann.moves.base import State

if TYPE_CHECKING:
    import josiann.typing as jot


# ====================================================
# code
class RandomStep(Move):
    """
    Simple random step within a radius of (-0.5 * magnitude) to (+0.5 * magnitude) around x.

    Args:
        magnitude: size of the random step is (-0.5 * magnitude) to (+0.5 * magnitude)
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self, magnitude: float, bounds: npt.NDArray[np.float64] | None = None):
        super().__init__(bounds=bounds)

        self._magnitude = magnitude

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
        target_dim = np.random.randint(len(x))
        increment = np.zeros(len(x), dtype=x.dtype)
        increment[target_dim] = self._magnitude * (np.random.random() - 0.5)

        return x + increment  # type: ignore[return-value]


class Metropolis(Move):
    """
    Metropolis step obtained from a multivariate normal distribution with mean <x> and covariance matrix <variances>

    Args:
        variances: list of variances between dimensions, which will be set as the diagonal of the covariance
            matrix.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(
        self,
        variances: npt.NDArray[np.float64],
        bounds: npt.NDArray[np.float64] | None = None,
    ):
        super().__init__(bounds=bounds)
        self._cov = np.diag(variances)

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
        return np.random.multivariate_normal(x, self._cov)  # type: ignore[return-value]


class Metropolis1D(Move):
    """
    Metropolis step obtained from a uni-variate normal distribution with mean <x> and variance <variance>

    Args:
        variance: the variance.
        bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
    """

    def __init__(self, variance: float, bounds: npt.NDArray[np.float64] | None = None):
        super().__init__(bounds=bounds)
        self._var = float(variance)

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
        target_dim = np.random.randint(len(x))
        x[target_dim] = np.random.normal(x[target_dim], self._var)

        return x
