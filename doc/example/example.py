# coding: utf-8
# Created on 18/01/2023 10:48
# Author : matteo

# ====================================================
# imports
import numpy as np

import numpy.typing as npt

from josiann import sa
from josiann.moves import Metropolis


# ====================================================
# code
def cost_function(x: npt.NDArray[np.float64]) -> float:
    return np.sum(x**2) + np.random.normal(0, 3)


def main():
    x0 = np.array(
        [
            [
                np.random.randint(-3, 4),  # random number in (-3, 3)
                np.random.choice(np.linspace(0.5, 5, 10)),
            ]
        ]
    )  # random number in (0.5, 5)

    res = sa(
        cost_function,
        x0,
        bounds=[(-3, 3), (0.5, 5)],
        moves=Metropolis(np.array([0.1, 0.1])),
        max_iter=200,
        max_measures=1000,
        T_0=5,
        seed=42,
    )

    res.trace.plot_parameters()
    res.trace.plot_positions(true_values=[0, 0.5])


if __name__ == "__main__":
    main()
