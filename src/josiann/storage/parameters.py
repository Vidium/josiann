from __future__ import annotations

import collections.abc
import numbers
from abc import ABC
from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
from attrs import frozen

import josiann.typing as jot
from josiann.backup.backup import Backup
from josiann.errors import ShapeError
from josiann.moves.base import Move
from josiann.moves.discrete import DiscreteMove


@frozen(kw_only=True)
class BaseParameters:
    """
    Object for storing the general parameters used for running the SA algorithm.
    """

    args: tuple[Any, ...]
    x0: npt.NDArray[Any]
    max_iter: int
    max_measures: int
    epsilon: float
    T_0: float
    tol: float
    alpha: float
    sigma_max: float
    suppress_warnings: bool
    detect_convergence: bool

    @property
    def x(self) -> npt.NDArray[Any]:
        return self.x0.copy()

    @property
    def nb_dimensions(self) -> int:
        return int(self.x0.shape[1])


@frozen(kw_only=True)
class MultiParameters(ABC):
    """
    Object for storing the parameters managing the calls to parallel or vectorized cost functions.
    """

    nb_walkers: int


@frozen(repr=False)
class MoveParameters:
    """
    Object for storing the moves and their probabilities.
    """

    list_probabilities: list[float]
    list_moves: Sequence[Move]

    def __repr__(self) -> str:
        return (
            "\t"
            + "\n\t".join(
                [
                    f"{type(move).__name__} with probability {np.round(proba, 4)}"
                    for move, proba in zip(self.list_moves, self.list_probabilities)
                ]
            )
            + "\n"
        )

    @property
    def using_SetMoves(self) -> bool:
        """
        Is at least one of the moves a Set move ?
        """
        return any([isinstance(move, DiscreteMove) for move in self.list_moves])

    def set_bounds(self, bounds: tuple[float, float] | Sequence[tuple[float, float]] | None) -> None:
        """
        Set the bounds for all moves.

        Args:
            bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
                (lower_bound, upper_bound)
                or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        """
        for move in self.list_moves:
            move.set_bounds(bounds)


@frozen(kw_only=True)
class SAParameters(ABC):
    """
    Object for storing the parameters used for running the SA algorithm.
    """

    base: BaseParameters
    multi: MultiParameters
    moves: MoveParameters
    fun: jot.FUN_TYPE[...] | jot.VECT_FUN_TYPE[...] | jot.PARALLEL_FUN_TYPE[...]
    backup: Backup[Any]
    costs: npt.NDArray[np.floating[Any]]
    last_ns: list[int]
    window_size: int
    seed: int


def check_base_parameters_core(
    T_0: float,
    epsilon: float,
    alpha: float,
    max_iter: int,
    max_measures: int,
    suppress_warnings: bool,
    tol: float,
    x0: npt.NDArray[jot.DType],
    dtype: jot.DType,
) -> tuple[float, float, int, int, float, npt.NDArray[jot.DType]]:
    # max iterations
    if max_iter < 0:
        raise ValueError("'max_iter' parameter must be positive.")

    # max function evaluations
    if max_measures < 0:
        raise ValueError("'max_measures' parameter must be positive.")

    # alpha
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError(f"Invalid value '{alpha}' for 'alpha', should be in [0, 1].")

    # epsilon
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"Invalid value '{epsilon}' for 'epsilon', should be in (0, 1).")

    # T 0
    if T_0 < 0:
        raise ValueError("'T_0' parameter must be at least 0.")

    # tolerance
    if tol <= 0:
        raise ValueError("'tol' parameter must be strictly positive.")

    # max sigma
    sigma_max = np.sqrt((max_measures - 1) * T_0 * alpha * (1 - epsilon)) / 3

    # suppress warnings
    if not suppress_warnings and max_iter < 200:
        warn("It is not recommended running the SA algorithm with less than 200 iterations.")

    return (
        float(T_0),
        alpha,
        int(max_iter),
        int(max_measures),
        sigma_max,
        x0.astype(dtype),
    )


def check_base_parameters(
    args: tuple[Any, ...] | None,
    x0: npt.NDArray[jot.DType],
    nb_walkers: int,
    max_iter: int,
    max_measures: int,
    alpha: float,
    epsilon: float,
    T_0: float,
    tol: float,
    suppress_warnings: bool,
    detect_convergence: bool,
    dtype: jot.DType,
) -> BaseParameters:
    """
    Check validity of base parameters.

    Args:
        args: an optional sequence of arguments to pass to the function to minimize.
        x0: a <d> dimensional vector of initial values.
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        alpha: cooling coefficient.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        dtype: the data type for the values stored in the Trace.

    Returns:
        BaseParameters.
    """
    # arguments
    args = tuple(args) if args is not None else ()

    # initial values
    if x0.ndim == 1:
        if len(x0) > 1:
            x0 = np.array([x0 + np.random.uniform(-0.5e-10, 0.5e-10) for _ in range(nb_walkers)])

        else:
            x0 = np.array([x0])

    if x0.shape[0] != nb_walkers:
        raise ShapeError(
            f"Matrix of initial values should have {nb_walkers} rows (equal to the number of "
            f"parallel walkers), not {x0.shape[0]}"
        )

    if len(x0) > 1 and np.all([x0[0] == x0[i] for i in range(1, len(x0))]):
        warn("Initial positions are the same for all walkers, adding random noise.")

        x0 = np.array([x0[i] + np.random.uniform(-0.5e-10, 0.5e-10) for i in range(len(x0))])

    (
        T_0,
        alpha,
        max_iter,
        max_measures,
        sigma_max,
        x0_casted,
    ) = check_base_parameters_core(
        T_0,
        epsilon,
        alpha,
        max_iter,
        max_measures,
        suppress_warnings,
        tol,
        x0,
        dtype,
    )

    return BaseParameters(
        args=args,
        x0=x0_casted,
        max_iter=max_iter,
        max_measures=max_measures,
        epsilon=epsilon,
        T_0=T_0,
        tol=tol,
        alpha=alpha,
        sigma_max=sigma_max,
        suppress_warnings=suppress_warnings,
        detect_convergence=detect_convergence,
    )


def check_bounds(
    bounds: tuple[float, float] | Sequence[tuple[float, float]] | None,
    x0: npt.NDArray[jot.DType],
) -> None:
    """
    Check validity of bounds.

    Args:
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        x0: a <d> dimensional vector of initial values or a matrix of initial values of shape (nb_walkers, d).
    """
    if bounds is not None:
        if (
            isinstance(bounds, tuple)
            and isinstance(bounds[0], numbers.Number)
            and isinstance(bounds[1], numbers.Number)
        ):
            if np.any(x0 < bounds[0]) or np.any(x0 > bounds[1]):  # type: ignore[operator]
                raise ValueError("Some values in x0 do not lie in between defined bounds.")

        elif isinstance(bounds, collections.abc.Sequence):
            if len(bounds) != x0.shape[1]:
                raise ShapeError(
                    f"Bounds must be defined for all dimensions, but only {len(bounds)} out of"
                    f" {x0.shape[1]} were defined."
                )

            for dim_index, bound in enumerate(bounds):
                if (
                    isinstance(bound, tuple)
                    and isinstance(bound[0], numbers.Number)
                    and isinstance(bound[1], numbers.Number)
                ):
                    if np.any(x0[:, dim_index] < bound[0]) or np.any(  # type: ignore[operator]
                        x0[:, dim_index] > bound[1]  # type: ignore[operator]
                    ):
                        raise ValueError(
                            f"Some values in x0 do not lie in between defined bounds for dimensions {dim_index}."
                        )

                else:
                    raise TypeError(
                        "'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                        "with the following format: \n"
                        "\t(lower_bound, upper_bound)\n "
                        "or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions."
                    )

        else:
            raise TypeError(
                "'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                "with the following format: \n"
                "\t(lower_bound, upper_bound)\n "
                "or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions."
            )
