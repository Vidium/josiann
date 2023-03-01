# coding: utf-8
# Created on 04/12/2022 17:56
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import Callable

from josiann import mcsa
from josiann import Result
from josiann.moves.base import Move
from josiann.moves.ensemble import Stretch


# ====================================================
# code
def run_sa(
    move: Move,
    cost_func: Callable,
    BOUNDS: list[tuple[float, float]],
    nb_walkers: int = 1,
    backup: bool = False,
    max_measures: int = 1000,
    max_iter: int = 200,
    nb_cores: int = 5,
) -> Result:
    seed = 42
    np.random.seed(seed)

    x0 = np.array(
        [
            [np.random.randint(-3, 4), np.random.choice(np.linspace(0.5, 5, 10))]
            for _ in range(nb_walkers)
        ]
    )

    res = mcsa(
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
        nb_cores=nb_cores,
        seed=seed,
    )

    assert res.parameters.backup.active == backup, print(res.parameters.backup.active)

    return res


# multi cores =================================================================
def test_multicore(BOUNDS, cost_function):
    print("Test parallel")
    res = run_sa(
        Stretch(bounds=BOUNDS),
        cost_func=cost_function,
        nb_walkers=5,
        nb_cores=5,
        BOUNDS=BOUNDS,
    )

    x = res.trace.positions.get_best().x

    assert np.allclose(x, [0, 0.5], atol=3e-1), res.message
