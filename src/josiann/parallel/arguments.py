from __future__ import annotations

from collections.abc import Generator, Iterator
from itertools import accumulate
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from attrs import define, field

_T = TypeVar("_T")
_DT = TypeVar("_DT", bound=np.generic)


def pairwise(iterable: Iterator[_T]) -> Generator[tuple[_T, _T], None, None]:
    try:
        successor = next(iterable)
    except StopIteration:
        pass
    else:
        for e in iterable:
            current, successor = successor, e
            yield current, successor


@define
class ParallelArgument(Generic[_DT]):
    """
    Object passed to parallel cost functions which holds instructions on what should be computed.

    Args:
        positions: matrix of position vectors at current iteration
        nb_evaluations: array indicating the number of evaluations to compute per position vector
        args: parallel arguments
    """

    positions: npt.NDArray[_DT]  #: matrix of position vectors at current iteration
    nb_evaluations: npt.NDArray[np.int_]  #: array indicating the number of evaluations to compute per position vector
    args: tuple[npt.NDArray[Any], ...] = field(factory=tuple)  #: parallel arguments
    _result: npt.NDArray[np.floating[Any]] = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._result = np.zeros(np.sum(self.nb_evaluations))

    @property
    def where_evaluations(self) -> tuple[npt.NDArray[Any], ...]:
        """
        Convenience attribute for getting formatted tuples of (position vector, parallel arguments ...) repeated as
        many times as required by `nb_evaluations`.
        """
        positions = np.repeat(self.positions, self.nb_evaluations, axis=0)

        args = tuple(np.repeat(arg, self.nb_evaluations)[:, np.newaxis] for arg in self.args)

        return (positions,) + args

    @property
    def result(self) -> npt.NDArray[np.floating[Any]]:
        """Array of costs of length equal to the total number of function evaluations required at current iteration."""
        return self._result

    @result.setter
    def result(self, result: npt.NDArray[np.floating[Any]]) -> None:
        self._result[:] = result

    def result_iter(self) -> Generator[npt.NDArray[np.floating[Any]], None, None]:
        for start, end in pairwise(accumulate(self.nb_evaluations, initial=0)):
            yield self._result[start:end]
