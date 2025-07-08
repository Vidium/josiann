# ====================================================
# imports
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.algorithms.map.typing import Execution
from josiann.algorithms.map.utils import _update_walker
from josiann.storage.parameters import SAParameters


# ====================================================
# code
def multicore_execution(max_workers: int) -> Execution:
    """
    Multicore executor for calling <fn> with parameters passed in <iterables> in parallel on multiple CPU cores.
    Equivalent to map(fn, *iter) for iter in iterables.

    Args:
        max_workers: max number of parallel workers.

    Returns:
        An executor for calling <fn> in parallel on multiple cores.
    """
    executor = ProcessPoolExecutor(max_workers=max_workers)

    def call(
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
        Executor for calling <fn> in parallel on multiple cores.

        Args:
            params: parameters of an SA run.
            x: current position vectors for the walkers to update of shape (nb_walkers,).
            costs: set of costs evaluated at last position vectors of shape (nb_walkers,).
            current_n: current required number of evaluations.
            last_ns: set of number of evaluations required when the last position vectors were accepted of shape
                (nb_walkers,).
            iteration: the current iteration number.
            temperature: the current temperature.
            kwargs:
                timeout: The maximum number of seconds to wait. If None, then there is no limit on the wait time.

        Returns:
            An iterator over map(fn, *iter).
        """
        complementary_sets = [np.delete(x.copy(), walker_index) for walker_index in range(len(x))]

        return executor.map(
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
            timeout=kwargs["timeout"],
        )

    return call
