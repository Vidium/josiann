# coding: utf-8
# Created on 03/02/2022 15:47
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np

from typing import Callable, Optional, Any

from josiann import sa, Result, Move, RandomStep, Metropolis, Metropolis1D, SetStep, Stretch, StretchAdaptive, \
    SetStretch


# ====================================================
# code
BOUNDS = [(-3, 3), (0.5, 5)]


def cost_function(x: np.ndarray) -> float:
    return np.sum(x ** 2) + np.random.normal(0, 3)


def vectorized_cost_function(x: np.ndarray) -> list[float]:
    return np.sum(x ** 2, axis=1) + np.random.normal(0, 1, size=len(x))


def vectorized_deterministic_cost_function(x: np.ndarray) -> list[float]:
    return list(np.sum(x ** 2, axis=1))


def run_sa(move: Move,
           cost_func: Callable,
           nb_walkers: int = 1,
           nb_cores: int = 1,
           vectorized: bool = False,
           backup: bool = False,
           nb_slots: Optional[int] = None,
           max_measures: int = 1000,
           max_iter: int = 200,
           vectorized_on_evaluations: bool = True,
           vectorized_skip_marker: Any = None) -> Result:
    seed = 42
    np.random.seed(seed)

    x0 = np.array([[np.random.randint(-3, 4), np.random.choice(np.linspace(0.5, 5, 10))] for _ in range(nb_walkers)])

    res = sa(cost_func,
             x0,
             bounds=BOUNDS,
             moves=move,
             nb_walkers=nb_walkers,
             max_iter=max_iter,
             max_measures=max_measures,
             final_acceptance_probability=1e-300,
             epsilon=0.001,
             T_0=5,
             tol=1e-3,
             nb_cores=nb_cores,
             vectorized=vectorized,
             vectorized_on_evaluations=vectorized_on_evaluations,
             vectorized_skip_marker=vectorized_skip_marker,
             backup=backup,
             nb_slots=nb_slots,
             seed=seed)

    assert res.parameters.parallel.nb_cores == nb_cores, print(res.parameters.parallel.nb_cores)
    assert res.parameters.parallel.vectorized == vectorized, print(res.parameters.parallel.vectorized)
    assert res.parameters.active_backup == backup, print(res.parameters.active_backup)

    # assert res.success, print(res)

    return res


res = run_sa(RandomStep(magnitude=0.5, bounds=BOUNDS),
             cost_func=cost_function)
