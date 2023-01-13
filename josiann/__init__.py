# coding: utf-8
# Created on 23/07/2021 23:29
# Author : matteo

"""
Just anOther SIMulated ANNealing.
This package provides an implementation of the simulated annealing method for minimizing noisy functions.
"""

# ====================================================
# imports
from josiann.sequential.base.sa import sa
from josiann.sequential.vectorized.vsa import vsa
from josiann.sequential.multicore.mcsa import mcsa
from josiann.parallel.psa import psa

from josiann.storage.result import Result
from josiann.storage.trace import Trace

# ====================================================
# code
