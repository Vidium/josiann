# coding: utf-8
# Created on 16/06/2022 09:29
# Author : matteo

# ====================================================
# imports
import numpy as np
import plotly.express as px

import josiann as jo


# ====================================================
# code
def vect_cost(x: np.ndarray,
              n: np.ndarray) -> np.ndarray:
    return 0.6 + np.sum([np.sin(1 - 16 / 15 * x[:, i]) ** (n+1) -
                         1 / 50 * np.sin(4 - 64 / 15 * x[:, i]) ** n -
                         np.sin(1 - 16 / 15 * x[:, i]) ** n
                         for i in range(x.shape[1])], axis=0)


def parallel_sa():
    res = jo.parallel_sa(vect_cost,
                         np.array([[0, 0], [0, 0], [0, 0]]),
                         parallel_args=[np.array([0, 1, 2])],
                         bounds=[(-1, 1), (-1, 1)],
                         moves=jo.ParallelSetStep(np.tile(np.linspace(-1, 1, 21), (2, 1))),
                         max_measures=1,
                         max_iter=2000,
                         backup=True,
                         seed=42,
                         detect_convergence=False)

    res.trace.plot_positions('/home/matteo/Desktop/test_positions.html',
                             true_values=np.array([[1, 1], [0.47, 0.47], [0.31, 0.31]]),
                             extended=True)

    fig = px.imshow(res.trace._cost_trace, aspect=3 / 200)
    fig.write_html('/home/matteo/Desktop/cost_trace.html')

    print('DONE')


if __name__ == '__main__':
    parallel_sa()
