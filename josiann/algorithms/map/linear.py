# coding: utf-8
# Created on 13/01/2023 10:21
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from itertools import repeat

import numpy.typing as npt
from typing import Any
from typing import Iterator

import josiann.typing as jot
from josiann.algorithms.map.utils import _update_walker
from josiann.storage.parameters import SAParameters


# ====================================================
# code
def linear_execution(
    params: SAParameters,
    x: npt.NDArray[jot.DType],
    costs: npt.NDArray[np.float64],
    current_n: int,
    last_ns: npt.NDArray[np.int64],
    iteration: int,
    temperature: float,
    **kwargs: Any,
) -> Iterator[tuple[npt.NDArray[jot.DType], float, bool, int]]:
    """
    Linear executor for calling <fn> with parameters passed in <iterables> in a linear fashion.
    Equivalent to map(fn, *iter) for iter in iterables.

    Args:
        params: parameters of an SA run.
        x: current position vectors for the walkers to update of shape (nb_walkers,).
        costs: set of costs evaluated at last position vectors of shape (nb_walkers,).
        current_n: current required number of evaluations.
        last_ns: set of number of evaluations required when the last position vectors were accepted of shape
            (nb_walkers,).
        iteration: the current iteration number.
        temperature: the current temperature.

    Returns:
        An iterator over map(fn, *iter).
    """
    complementary_sets = [
        np.delete(x.copy(), walker_index) for walker_index in range(len(x))
    ]

    return map(
        _update_walker,
        repeat(params.fun),
        x,
        costs,
        repeat(current_n),
        last_ns,
        repeat(params.base.args),
        repeat(params.moves.list_moves),
        repeat(params.moves.list_probabilities),
        repeat(iteration),
        repeat(params.base.max_iter),
        repeat(temperature),
        repeat(params.backup),
        complementary_sets,
        range(len(x)),
    )
