# coding: utf-8
# Created on 24/09/2021 14:33
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import Callable

from josiann import sa
from josiann import Result
from josiann.moves.base import Move
from josiann.moves.sequential import RandomStep
from josiann.moves.sequential import Metropolis
from josiann.moves.sequential import Metropolis1D
from josiann.moves.set import SetStep
from josiann.moves.set import SetStretch
from josiann.moves.ensemble import Stretch
from josiann.moves.ensemble import StretchAdaptive


# ====================================================
# code
BOUNDS = [(-3, 3), (0.5, 5)]


def cost_function(x: np.ndarray) -> float:
    return np.sum(x**2) + np.random.normal(0, 3)


def run_sa(
    move: Move,
    cost_func: Callable,
    nb_walkers: int = 1,
    backup: bool = False,
    max_measures: int = 1000,
    max_iter: int = 200,
) -> Result:
    seed = 42
    np.random.seed(seed)

    x0 = np.array(
        [
            [np.random.randint(-3, 4), np.random.choice(np.linspace(0.5, 5, 10))]
            for _ in range(nb_walkers)
        ]
    )

    res = sa(
        cost_func,
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
        backup=backup,
        seed=seed,
    )

    assert res.parameters.backup.active == backup, print(res.parameters.backup.active)

    return res


# ONE walker ------------------------------------------------------------------
def test_RandomStep():
    res = run_sa(RandomStep(magnitude=5, bounds=BOUNDS), cost_func=cost_function)

    assert np.allclose(res.trace.positions.get_best().x, [0, 0.5], atol=3e-1)


def test_Metropolis():
    res = run_sa(
        Metropolis(variances=np.array([0.2, 0.2]), bounds=BOUNDS),
        cost_func=cost_function,
    )

    assert np.allclose(res.trace.positions.get_best().x, [0, 0.5], atol=3e-1)


def test_Metropolis1D():
    res = run_sa(Metropolis1D(variance=0.2, bounds=BOUNDS), cost_func=cost_function)

    assert np.allclose(res.trace.positions.get_best().x, [0, 0.5], atol=3e-1)


def test_SetStep():
    res = run_sa(
        SetStep(
            position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
            bounds=BOUNDS,
        ),
        cost_func=cost_function,
    )

    x = res.trace.positions.get_best().x[0]

    assert x[0] in [-0.25, 0, 0.25] and x[1] in [0.5, 0.75]


# MULTIPLE walkers ------------------------------------------------------------
def test_Stretch():
    res = run_sa(Stretch(bounds=BOUNDS), cost_func=cost_function, nb_walkers=5)

    assert np.allclose(res.trace.positions.get_best().x, [0, 0.5], atol=3e-1)


def test_StretchAdaptive():
    res = run_sa(
        StretchAdaptive(a=3, bounds=BOUNDS), cost_func=cost_function, nb_walkers=5
    )

    assert np.allclose(res.trace.positions.get_best().x, [0, 0.5], atol=3e-1)


def test_SetStretch():
    res = run_sa(
        SetStretch(
            position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
            bounds=BOUNDS,
        ),
        cost_func=cost_function,
        nb_walkers=5,
    )

    x = res.trace.positions.get_best().x

    assert np.all(np.in1d(x[:, 0], [-0.25, 0, 0.25]))
    assert np.all(np.in1d(x[:, 1], [0.5, 0.75]))
