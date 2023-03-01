# coding: utf-8

# ====================================================
# imports
import pytest
import numpy as np

import numpy.typing as npt
from typing import Any


# ====================================================
# code
@pytest.fixture
def BOUNDS():
    return [(-3, 3), (0.5, 5)]


def _cost_function(x: npt.NDArray[Any]) -> float:
    return np.sum(x**2) + np.random.normal(0, 3)


@pytest.fixture
def cost_function():
    return _cost_function
