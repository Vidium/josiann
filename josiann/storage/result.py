# coding: utf-8
# Created on 03/02/2022 11:54
# Author : matteo

# ====================================================
# imports
from attrs import frozen

from .trace import Trace
from .parameters import SAParameters


# ====================================================
# code
@frozen
class Result:
    """
    Object for storing the results of a run.
    """
    message: str
    success: bool
    trace: Trace
    parameters: SAParameters
