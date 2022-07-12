# coding: utf-8
# Created on 23/07/2021 23:29
# Author : matteo

"""
Just anOther SIMulated ANNealing.
This package provides an implementation of the simulated annealing method for minimizing noisy functions.
"""

# ====================================================
# imports
from .sa import sa
from .moves import Move, RandomStep, Metropolis, Metropolis1D, Stretch, StretchAdaptive, SetStep, SetStretch
from .storage import Result, Trace
from .parallel import parallel_sa, ParallelMove, ParallelSetStep

# ====================================================
# code
__all__ = ['sa', 'parallel_sa',
           'Move', 'ParallelMove',
           'RandomStep', 'Metropolis', 'Metropolis1D', 'SetStep', 'Stretch', 'StretchAdaptive', 'SetStretch',
           'ParallelSetStep',
           'Trace', 'Result']
