# coding: utf-8
# Created on 23/07/2021 23:29
# Author : matteo

"""
Just anOther SIMulated ANNealing.
This package provides an implementation of the simulated annealing method for minimizing noisy functions.
"""

# ====================================================
# imports
try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata  # type: ignore[import, no-redef]

from josiann.algorithms.sequential.base.sa import sa
from josiann.algorithms.sequential.multicore.mcsa import mcsa
from josiann.algorithms.sequential.vectorized.vsa import vsa
from josiann.moves.discrete import SetStep, SetStretch
from josiann.moves.ensemble import Stretch, StretchAdaptive
from josiann.moves.sequential import Metropolis, Metropolis1D, RandomStep
from josiann.storage.result import Result
from josiann.storage.trace import Trace

# ====================================================
# code
__all__ = [
    "sa",
    "vsa",
    "mcsa",
    "Result",
    "Trace",
    "RandomStep",
    "Metropolis",
    "Metropolis1D",
    "SetStep",
    "SetStretch",
    "Stretch",
    "StretchAdaptive",
]

__version__ = metadata.version("josiann")
