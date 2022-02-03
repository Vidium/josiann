# coding: utf-8
# Created on 24/09/2021 14:33
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


# ONE walker ------------------------------------------------------------------
def test_RandomStep():
    print('Test RandomStep')
    res = run_sa(RandomStep(magnitude=5,
                            bounds=BOUNDS),
                 cost_func=cost_function)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), res.x


def test_Metropolis():
    print('Test Metropolis')
    res = run_sa(Metropolis(variances=np.array([0.2, 0.2]),
                            bounds=BOUNDS),
                 cost_func=cost_function)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), res.x


def test_Metropolis1D():
    print('Test Metropolis1D')
    res = run_sa(Metropolis1D(variance=0.2,
                              bounds=BOUNDS),
                 cost_func=cost_function)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), res.x


def test_SetStep():
    print('Test SetStep')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=cost_function)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


# MULTIPLE walkers ------------------------------------------------------------
def test_Stretch():
    print('Test Stretch')
    res = run_sa(Stretch(bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), res.x


def test_StretchAdaptive():
    print('Test StretchAdaptive')
    res = run_sa(StretchAdaptive(a=3, bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), res.x


def test_SetStretch():
    print('Test SetStretch')
    res = run_sa(SetStretch(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                            bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


# multi cores =================================================================
# REMINDER : this must be launched in terminal as 'python -m tests.test_sa' to get through the if __name__ == '__main__'
@pytest.mark.multicores
def test_parallel():
    print('Test parallel')
    res = run_sa(Stretch(bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5,
                 nb_cores=5)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), res.x


# vectorized ==================================================================
def test_vectorized():
    print('Test vectorized')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=vectorized_cost_function,
                 nb_walkers=5,
                 vectorized=True)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


def test_vectorized_on_walkers():
    print('Test vectorized on walkers')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=vectorized_cost_function,
                 nb_walkers=5,
                 vectorized=True,
                 vectorized_on_evaluations=False,
                 vectorized_skip_marker=np.array([0., 0.]))

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


def test_vectorized_on_walkers_and_slots():
    print('Test vectorized on walkers and slots')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=vectorized_cost_function,
                 nb_walkers=1,
                 vectorized=True,
                 vectorized_on_evaluations=False,
                 vectorized_skip_marker=np.array([1., 1.]),
                 nb_slots=4)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


# with backup =================================================================
def test_backup():
    print('Test backup')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=vectorized_cost_function,
                 nb_walkers=5,
                 vectorized=True,
                 backup=True)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


# multi slots =================================================================
def test_slots():
    print('Test slots')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=vectorized_cost_function,
                 nb_walkers=5,
                 vectorized=True,
                 backup=True,
                 nb_slots=50)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


# cost function deterministic =================================================================
def test_deterministic():
    print('Test deterministic')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=vectorized_deterministic_cost_function,
                 nb_walkers=5,
                 vectorized=True,
                 backup=True,
                 max_measures=1)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], res.x


if __name__ == '__main__':
    # test_RandomStep()
    test_Metropolis()
    test_Metropolis1D()
    test_SetStep()
    test_Stretch()
    test_StretchAdaptive()
    test_SetStretch()
    test_parallel()
    test_vectorized()
    test_vectorized_on_walkers()
    test_vectorized_on_walkers_and_slots()
    test_backup()
    test_slots()
    test_deterministic()
