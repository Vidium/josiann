# coding: utf-8
# Created on 04/12/2022 17:56
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np

from typing import Callable

from josiann import mcsa
from josiann import Result
from josiann.moves.base import Move
from josiann.moves.ensemble import Stretch

from .test_sa import BOUNDS, cost_function


# ====================================================
# code
def run_sa(
    move: Move,
    cost_func: Callable,
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

    print(res.message)

    assert res.parameters.backup.active == backup, print(res.parameters.backup.active)

    return res


# multi cores =================================================================
# REMINDER : this must be launched in terminal as 'python -m pytest tests/test_mcsa.py' to get through the if __name__
# == '__main__'
@pytest.mark.multicores
def test_multicore():
    print("Test parallel")
    res = run_sa(
        Stretch(bounds=BOUNDS), cost_func=cost_function, nb_walkers=5, nb_cores=5
    )

    x = res.trace.positions.get_best().x

    assert np.allclose(x, [0, 0.5], atol=3e-1)


if __name__ == "__main__":
    test_multicore()
