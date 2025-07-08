"""
Core Simulated Annealing function.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm.autonotebook import tqdm

from josiann.algorithms.map.typing import Execution
from josiann.compute import T, n, sigma
from josiann.storage.parameters import SAParameters
from josiann.storage.result import Result
from josiann.storage.trace import Trace


def run_simulated_annealing(
    costs: npt.NDArray[np.float64],
    execution: Execution,
    last_ns: npt.NDArray[np.int64],
    nb_walkers: int,
    params: SAParameters,
    progress_bar: range | tqdm[int],
    trace: Trace,
    x: npt.NDArray[np.float64 | np.int64],
    **kwargs: Any,
) -> Result:
    iteration = 0

    for iteration in progress_bar:
        temperature = T(iteration, params.base.T_0, params.base.alpha)
        current_n = n(iteration, params.base)
        current_sigma = sigma(iteration, params.base.T_0, params.base.alpha, params.base.epsilon)

        accepted = np.zeros(params.multi.nb_walkers, dtype=bool)
        rescued = np.zeros(params.multi.nb_walkers, dtype=bool)
        explored = np.zeros((params.multi.nb_walkers, params.base.nb_dimensions))
        explored_costs = np.zeros(params.multi.nb_walkers)

        acceptance_fraction = trace.positions.mean_acceptance_fraction(iteration)

        start = time.perf_counter()

        try:
            updates = execution(
                params,
                x=x.copy(),
                costs=costs,
                current_n=current_n,
                last_ns=last_ns,
                iteration=iteration,
                temperature=temperature,
                **kwargs | dict(acceptance=acceptance_fraction if acceptance_fraction is not np.nan else 1.0),
            )

            for _x, _cost, _accepted, _walker_index in updates:
                if _accepted:
                    # update positions, costs, n
                    x[_walker_index] = _x
                    costs[_walker_index] = _cost
                    last_ns[_walker_index] = current_n

                accepted[_walker_index] = _accepted
                explored[_walker_index] = _x
                explored_costs[_walker_index] = _cost

        except Exception:
            message = f"Unexpected failure while evaluating cost function : \n{traceback.format_exc()}"
            success = False
            break

        elapsed = time.perf_counter() - start

        trace.positions.store(iteration, x, np.array(costs), current_n, accepted, explored, explored_costs)
        trace.parameters.store(iteration, temperature, current_n, current_sigma, elapsed)

        stuck_walkers = trace.positions.are_stuck(iteration)
        best = trace.positions.get_best(iteration)
        best_index = np.argmin(best.cost)

        for _walker_index in range(nb_walkers):
            # rescue stuck walkers
            if stuck_walkers[_walker_index]:
                x[_walker_index] = best.x[best_index]
                costs[_walker_index] = best.cost[best_index]
                last_ns[_walker_index] = best.n[best_index]
                rescued[_walker_index] = True

        trace.positions.update(iteration, x, costs, last_ns, rescued)

        if isinstance(progress_bar, tqdm):
            progress_bar.set_description(
                f"T: {temperature:.4f}"
                f"  A: {acceptance_fraction:.2%}%"
                f"  Best: {np.min(best.cost):.4f}"
                f"  Current: {np.min(costs):.4f}"
            )

        if np.all(trace.positions.converged):
            message, success = "Convergence tolerance reached.", True
            break

    else:
        message, success = "Requested number of iterations reached.", False

    trace.finalize(iteration)

    return Result(message, success, trace, params)


def restart(
    previous: Result,
    max_iter: int = 200,
    max_measures: int = 20,
    epsilon: float = 0.01,
    T_0: float = 5.0,
    tol: float = 1e-3,
    verbose: bool = True,
    suppress_warnings: bool = False,
    detect_convergence: bool = True,
    window_size: int | None = None,
) -> Result:
    """


    Args:
        previous: TODO
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        verbose: print progress bar ? (default True)
        suppress_warnings: remove warnings ? (default False)
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.

    Returns:

    """
    raise NotImplementedError
