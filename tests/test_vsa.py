# coding: utf-8
# Created on 04/12/2022 16:41
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import Any
from typing import Callable

from josiann import Result
from josiann import vsa
from josiann.moves import Move
from josiann.moves import SetStep

# ====================================================
# code
BOUNDS = [(-3, 3), (0.5, 5)]


def vectorized_cost_function(x: np.ndarray) -> list[float]:
    return np.sum(x ** 2, axis=1) + np.random.normal(0, 1, size=len(x))


def vectorized_deterministic_cost_function(x: np.ndarray) -> list[float]:
    return list(np.sum(x ** 2, axis=1))


def run_vsa(move: Move,
            cost_func: Callable,
            nb_walkers: int = 1,
            backup: bool = False,
            max_measures: int = 1000,
            max_iter: int = 200,
            vectorized_on_evaluations: bool = True,
            vectorized_skip_marker: Any = None,
            nb_slots: int | None = None) -> Result:
    seed = 42
    np.random.seed(seed)

    x0 = np.array([[np.random.randint(-3, 4), np.random.choice(np.linspace(0.5, 5, 10))] for _ in range(nb_walkers)])

    res = vsa(cost_func,
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
              vectorized_on_evaluations=vectorized_on_evaluations,
              vectorized_skip_marker=vectorized_skip_marker,
              backup=backup,
              nb_slots=nb_slots,
              seed=seed)

    assert res.parameters.backup.active == backup, print(res.parameters.backup.active)

    # assert res.success, print(res)

    return res


# vectorized ==================================================================
def test_vectorized():
    print('Test vectorized')
    res = run_vsa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                          bounds=BOUNDS),
                  cost_func=vectorized_cost_function,
                  nb_walkers=5)

    best = res.trace.positions.get_best_all()

    assert best.x[0] in [-0.25, 0, 0.25] and best.x[1] in [0.5, 0.75], best.x


def test_vectorized_on_walkers():
    print('Test vectorized on walkers')
    res = run_vsa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                          bounds=BOUNDS),
                  cost_func=vectorized_cost_function,
                  nb_walkers=5,
                  vectorized_on_evaluations=False,
                  vectorized_skip_marker=np.array([0., 0.]))

    best = res.trace.positions.get_best_all()

    assert best.x[0] in [-0.25, 0, 0.25] and best.x[1] in [0.5, 0.75], best.x


def test_vectorized_on_walkers_and_slots():
    print('Test vectorized on walkers and slots')
    res = run_vsa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                          bounds=BOUNDS),
                  cost_func=vectorized_cost_function,
                  nb_walkers=1,
                  vectorized_on_evaluations=False,
                  vectorized_skip_marker=np.array([1., 1.]),
                  nb_slots=4)

    best = res.trace.positions.get_best_all()

    assert best.x[0] in [-0.25, 0, 0.25] and best.x[1] in [0.5, 0.75], best.x


# with backup =================================================================
def test_backup():
    print('Test backup')
    res = run_vsa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                          bounds=BOUNDS),
                  cost_func=vectorized_cost_function,
                  nb_walkers=5,
                  backup=True)

    best = res.trace.positions.get_best_all()

    assert best.x[0] in [-0.25, 0, 0.25] and best.x[1] in [0.5, 0.75], best.x


# multi slots =================================================================
def test_slots():
    print('Test slots')
    res = run_vsa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                          bounds=BOUNDS),
                  cost_func=vectorized_cost_function,
                  nb_walkers=5,
                  backup=True,
                  nb_slots=50)

    best = res.trace.positions.get_best_all()

    assert best.x[0] in [-0.25, 0, 0.25] and best.x[1] in [0.5, 0.75], best.x


# cost function deterministic =================================================================
def test_deterministic():
    print('Test deterministic')
    res = run_vsa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                          bounds=BOUNDS),
                  cost_func=vectorized_deterministic_cost_function,
                  nb_walkers=5,
                  backup=True,
                  max_measures=1)

    best = res.trace.positions.get_best_all()

    assert best.x[0] in [-0.25, 0, 0.25] and best.x[1] in [0.5, 0.75], best.x
