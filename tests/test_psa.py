# coding: utf-8
# Created on 16/06/2022 09:29
# Author : matteo

# ====================================================
# imports
import numpy as np

from josiann.parallel import psa
from josiann.parallel.moves import ParallelSetStep


# ====================================================
# code
def vect_cost(x: np.ndarray,
              _converged: np.ndarray,
              n: np.ndarray) -> np.ndarray:
    return 0.6 + np.sum([np.sin(1 - 16 / 15 * x[:, i]) ** (n + 1) -
                         1 / 50 * np.sin(4 - 64 / 15 * x[:, i]) ** n -
                         np.sin(1 - 16 / 15 * x[:, i]) ** n
                         for i in range(x.shape[1])], axis=0)


def test_parallel_sa():
    res = psa(vect_cost,
              np.array([[0, 0], [0, 0], [0, 0]]),
              parallel_args=[np.array([0, 1, 2])],
              bounds=[(-1, 1), (-1, 1)],
              moves=ParallelSetStep(np.tile(np.linspace(-1, 1, 21), (2, 1))),
              max_measures=1,
              max_iter=2000,
              backup=True,
              seed=42,
              detect_convergence=False)

    assert np.allclose(res.trace.positions.get_best().x, np.array([[1., 1.],
                                                                   [0.5, 0.5],
                                                                   [0.3, 0.3]]))
