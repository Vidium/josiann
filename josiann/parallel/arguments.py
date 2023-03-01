# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from attrs import field
from attrs import define
from itertools import accumulate
from more_itertools import pairwise

import numpy.typing as npt
from typing import Any
from typing import Generator


# ====================================================
# code
@define
class ParallelArgument:
    """Object passed to parallel cost functions which holds instructions on what should be computed."""

    positions: npt.NDArray[np.float_ | np.int_]
    nb_evaluations: npt.NDArray[np.int_]
    args: tuple[npt.NDArray[Any], ...] = field(factory=tuple)
    _result: npt.NDArray[np.float_] = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._result = np.zeros(np.sum(self.nb_evaluations))

    @property
    def where_evaluations(self) -> tuple[npt.NDArray[Any], ...]:
        positions = np.repeat(self.positions, self.nb_evaluations, axis=0)

        args = tuple(
            np.repeat(arg, self.nb_evaluations)[:, np.newaxis] for arg in self.args
        )

        return (positions,) + args

    @property
    def result(self) -> npt.NDArray[np.float_]:
        return self._result

    @result.setter
    def result(self, result: npt.NDArray[np.float_]) -> None:
        self._result[:] = result

    def result_iter(self) -> Generator[npt.NDArray[np.float_], None, None]:
        for start, end in pairwise(accumulate(self.nb_evaluations, initial=0)):
            yield self._result[start:end]
