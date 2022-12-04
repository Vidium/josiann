# coding: utf-8
# Created on 23/07/2021 23:29
# Author : matteo

"""
Just anOther SIMulated ANNealing.
This package provides an implementation of the simulated annealing method for minimizing noisy functions.
"""

# ====================================================
# imports
from josiann.sequential.sa import sa            # noqa: F401
from josiann.vectorized.sa import vsa           # noqa: F401
from josiann.parallel.sa import psa             # noqa: F401

from josiann.storage.result import Result       # noqa: F401

# ====================================================
# code
