# coding: utf-8
# Created on 04/12/2022 17:56
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np

from josiann.moves import Stretch

from .test_sa import BOUNDS, cost_function, run_sa


# ====================================================
# code
# multi cores =================================================================
# REMINDER : this must be launched in terminal as 'python -m tests.test_sa' to get through the if __name__ == '__main__'
@pytest.mark.multicores
def test_parallel():
    print('Test parallel')
    res = run_sa(Stretch(bounds=BOUNDS),
                 cost_func=cost_function,
                 nb_walkers=5,
                 nb_cores=5)

    x = res.trace.positions.get_best()

    print(res)

    assert np.allclose(x, [0, 0.5], atol=3e-1)
