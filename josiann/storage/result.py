# coding: utf-8
# Created on 03/02/2022 11:54
# Author : matteo

# ====================================================
# imports
import numpy as np
from dataclasses import dataclass, field

from .trace import Trace
from .parameters import SAParameters


# ====================================================
# code
@dataclass
class Result:
    """
    Object for storing the results of a run.
    """
    message: str
    success: bool
    trace: Trace
    parameters: SAParameters
    x: np.ndarray = field(init=False)
    x_cost: float = field(init=False)
    x_iter: list[int] = field(init=False)

    def __post_init__(self):
        self.x, self.x_cost, self.x_iter = self.trace.get_best()

    def __repr__(self) -> str:
        return f"Message : {self.message}\n" \
               f"Success : {self.success}\n" \
               f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter[0] + 1} by walker " \
               f"#{self.x_iter[1]})\n" \
               f"x: {self.x}"

    def full_repr(self) -> None:
        print(f"Message : {self.message}\n"
              f"Success : {self.success}\n"
              f"Parameters : \n{self.parameters}\n"
              f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter[0] + 1} by walker "
              f"#{self.x_iter[1]})\n"
              f"x: {self.x}")
