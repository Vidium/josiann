"""
Just anOther SIMulated ANNealing.
This package provides an implementation of the simulated annealing method for minimizing noisy functions.
"""

from importlib import metadata

from josiann.algorithms.sequential.base.sa import sa
from josiann.algorithms.sequential.multicore.mcsa import mcsa
from josiann.algorithms.sequential.vectorized.vsa import vsa
from josiann.moves.discrete import SetStep, SetStretch
from josiann.moves.ensemble import Stretch, StretchAdaptive
from josiann.moves.sequential import Metropolis, Metropolis1D, RandomStep
from josiann.storage.result import Result
from josiann.storage.trace import Trace

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
