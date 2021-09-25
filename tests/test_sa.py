# coding: utf-8
# Created on 24/09/2021 14:33
# Author : matteo

# ====================================================
# imports
import numpy as np

from typing import List, Callable

from josiann import sa, Result, Move, RandomStep, Metropolis, Metropolis1D, SetStep, Stretch, StretchAdaptive, \
    SetStretch


# ====================================================
# code
BOUNDS = [(-3, 3), (0.5, 5)]


def cost_function(x: np.ndarray) -> float:
    return np.sum(x ** 2) + np.random.normal(0, 3)


def vectorized_cost_function(x: np.ndarray) -> List[float]:
    return np.sum(x ** 2, axis=1) + np.random.normal(0, 1, size=len(x))


def run_sa(move: Move,
           cost_func: Callable,
           nb_walkers: int = 1,
           nb_cores: int = 1,
           vectorized: bool = False,
           backup: bool = False) -> Result:
    seed = 42
    np.random.seed(seed)

    x0 = np.array([[np.random.randint(-3, 4), np.random.choice(np.linspace(0.5, 5, 10))] for _ in range(nb_walkers)])

    res = sa(cost_func,
             x0,
             bounds=BOUNDS,
             moves=move,
             nb_walkers=nb_walkers,
             max_iter=200,
             max_measures=20000,
             final_acceptance_probability=1e-300,
             epsilon=0.001,
             T_0=5,
             tol=1e-3,
             nb_cores=nb_cores,
             vectorized=vectorized,
             backup=backup,
             seed=seed)

    assert res.nb_cores == nb_cores, print(res.nb_cores)
    assert res.vectorized == vectorized, print(res.vectorized)
    assert res.active_backup == backup, print(res.active_backup)

    assert res.success, print(res)

    return res


# ONE walker ------------------------------------------------------------------
def test_RandomStep():
    print('Test RandomStep')
    res = run_sa(RandomStep(magnitude=1,
                            bounds=BOUNDS),
                 cost_func=cost_function)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), print(res.x)


def test_Metropolis():
    print('Test Metropolis')
    res = run_sa(Metropolis(variances=np.array([0.1, 0.1]),
                            bounds=BOUNDS),
                 cost_func=cost_function)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), print(res.x)


def test_Metropolis1D():
    print('Test Metropolis1D')
    res = run_sa(Metropolis1D(variance=1,
                              bounds=BOUNDS),
                 cost_func=cost_function)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), print(res.x)


def test_SetStep():
    print('Test SetStep')
    res = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                         bounds=BOUNDS),
                 cost_func=cost_function)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], print(res.x)


# MULTIPLE walkers ------------------------------------------------------------
def test_Stretch():
    print('Test Stretch')
    res = run_sa(Stretch(bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), print(res.x)


def test_StretchAdaptive():
    print('Test StretchAdaptive')
    res = run_sa(StretchAdaptive(a=3, bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), print(res.x)


def test_SetStretch():
    print('Test SetStretch')
    res = run_sa(SetStretch(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                            bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5)

    assert res.x[0] in [-0.25, 0, 0.25] and res.x[1] in [0.5, 0.75], print(res.x)


# multi cores =================================================================
def test_parallel():
    print('Test parallel')
    res = run_sa(Stretch(bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5,
                 nb_cores=5)

    assert np.allclose(res.x, [0, 0.5], atol=3e-1), print(res.x)


# vectorized ==================================================================
def test_vectorized():
    print('Test vectorized')
    _ = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                       bounds=BOUNDS),
               cost_func=vectorized_cost_function,
               nb_walkers=5,
               vectorized=True)


# with backup =================================================================
def test_backup():
    print('Test backup')
    _ = run_sa(SetStep(position_set=[np.linspace(-3, 3, 25), np.linspace(0.5, 5, 19)],
                       bounds=BOUNDS),
               cost_func=cost_function,
               nb_walkers=5,
               backup=True)


if __name__ == '__main__':
    test_RandomStep()
    test_Metropolis()
    test_Metropolis1D()
    test_SetStep()
    test_Stretch()
    test_StretchAdaptive()
    test_SetStretch()
    test_parallel()
    test_vectorized()
    test_backup()
