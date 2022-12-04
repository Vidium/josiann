# coding: utf-8
# Created on 04/12/2022 10:46
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import Any
from typing import Protocol
from typing import Iterator
from typing import Callable
from typing import Sequence
from typing import ParamSpec
from typing import Concatenate

from josiann.backup import Backup
from josiann.moves import Move

# ====================================================
# code
_P = ParamSpec('_P')

FUN_TYPE = Callable[Concatenate[np.ndarray, _P], float]                         # type: ignore
VECT_FUN_TYPE = Callable[Concatenate[np.ndarray, _P], list[float]]              # type: ignore

SA_UPDATE = tuple[np.ndarray | float, float, bool, int]


class Execution(Protocol):
    def __call__(self,
                 fn: FUN_TYPE | VECT_FUN_TYPE,
                 x: np.ndarray,
                 costs: np.ndarray,
                 current_n: int,
                 last_ns: np.ndarray,
                 args: tuple[Any, ...],
                 list_moves: Sequence[Move],
                 list_probabilities: list[float],
                 iteration: int,
                 max_iter: int,
                 temperature: float,
                 backup_storage: Backup,
                 **kwargs: Any) -> Iterator[SA_UPDATE]: ...
