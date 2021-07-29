# coding: utf-8
# Created on 23/07/2021 23:28
# Author : matteo

# ====================================================
# imports
import numpy as np
from abc import ABC

from typing import Optional, Sequence, Tuple, Union


# ====================================================
# code
class Move(ABC):
    """
    Base abstract class for
    """

    def __init__(self, bounds: Optional[Sequence[Tuple[float, float]]] = None):
        """
        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension.
        """
        self.__bounds = np.array(bounds) if bounds is not None else None

    def set_bounds(self, bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]]) -> None:
        """
        Set bounds for the Move.

        :param bounds: optional sequence of (min, max) bounds for values to propose in each dimension or a single
            (min, max) tuple of bounds to set for all dimensions.
        """
        self.__bounds = np.array(bounds) if bounds is not None else None

    def _valid_proposal(self, x: np.ndarray) -> np.ndarray:
        """
        Get valid proposal within defined bounds.

        :param x: a 'raw' proposal.
        :return: a proposal with values restricted with the defined bounds.
        """
        if self.__bounds is not None:
            return np.minimum(np.maximum(x, self.__bounds[:, 0]), self.__bounds[:, 1])

        return x
