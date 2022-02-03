# coding: utf-8
# Created on 23/07/2021 23:28
# Author : matteo

"""
Core Simulated Annealing function.
"""

# ====================================================
# imports
import time
import numpy as np
from tqdm import tqdm
from itertools import repeat

from typing import Callable, Optional, Sequence, Union, Any, Type

from .utils import n, T, sigma
from .moves import Move, RandomStep
from .storage import Trace, Result, initialize_sa
from .__mappers import LinearExecutor, VectorizedExecutor, ParallelExecutor


# ====================================================
# code
def sa(fun: Callable[[np.ndarray, Any], Union[float, list[float]]],
       x0: np.ndarray,
       args: Optional[Sequence] = None,
       bounds: Optional[Sequence[tuple[float, float]]] = None,
       moves: Union[Move, Sequence[Move], Sequence[tuple[float, Move]]] = ((0.8, RandomStep(0.05)),
                                                                           (0.2, RandomStep(0.5))),
       nb_walkers: int = 1,
       max_iter: int = 200,
       max_measures: int = 20,
       final_acceptance_probability: float = 1e-300,
       epsilon: float = 0.01,
       T_0: float = 5.,
       tol: float = 1e-3,
       nb_cores: int = 1,
       vectorized: bool = False,
       vectorized_on_evaluations: bool = True,
       vectorized_skip_marker: Any = None,
       backup: bool = False,
       nb_slots: Optional[int] = None,
       seed: Optional[int] = None,
       verbose: bool = True,
       suppress_warnings: bool = False,
       detect_convergence: bool = True,
       window_size: Optional[int] = None) -> Result:
    """
    Simulated Annealing for minimizing noisy cost functions.

    Args:
        fun: a <d> dimensional (noisy) function to minimize.
        x0: a <d> dimensional vector of initial values.
        args: an optional sequence of arguments to pass to the function to minimize.
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        moves: either
                    - a single josiann.SingleMove object
                    - a sequence of josiann.SingleMove objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.SingleMove)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        nb_cores: number of cores that can be used to move walkers in parallel.
        vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
            just one. (<nb_cores> parameter will be set to 1 in this case.)
        vectorized_on_evaluations: when using a vectorized function, the vectorization can happen on walkers or
            on function evaluations.
             - On function evaluations, a loop on walkers calls <fun> with a vector of positions of the walker, repeated
             for the number of needed function evaluations.
             Ex: 2 walkers with position vectors p1 and p2 each need n1 and n2 function evaluations. <fun> is first
             called for walker 1 with a vector (p1, p1, ..., p1) of size n1, then <fun> is called for walker 2 with a
             vector (p2, p2, ..., p2) of size n2.
             This is the default option and is valid when <fun> is ok with receiving vectors of varying length and when
             <max_measures> is greater than <nb_walkers>.

             - On walkers, a loop on function evaluations calls <fun> with a vector of fixed size = <nb_walkers>.
             Ex: 2 walkers with position vectors p1 and p2 each need n1 and n2 function evaluations. <fun> is called
             with vector (p1, p2) for max(n1, n2) times.
             Often, n1 =/= n2 which would yield unnecessary function evaluations (e.g. when n1 < n2, some evaluations of
             p1 are not needed while p2 is still evaluated). To indicated that to <fun>, the <vectorized_skip_marker> is
             passed instead of unnecessary position vectors (e.g. when n1 < n2, the vector passed to <fun> will
             eventually be (<vectorized_skip_marker>, p2) instead of (p1, p2)).
             This is valid when <fun> needs to receive vectors of fixed length and when <nb_walkers> is greater than
             <max_measures>.
        vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
            evaluation for a particular position vector can be skipped.
        backup: use Backup for storing previously computed function evaluations and reusing them when returning to
            the same position vector ? (Only available when using SetStep moves).
        nb_slots: When using a vectorized function, the total number of position vectors for which the cost can be
            computed at once.
            For example, when using 5 walkers and 22 slots, each walker will be attributed respectively 5, 5, 4, 4,
            and 4 slots.
            Multiple slots per walker are used for exploring multiple possible moves from the starting position vector
            of each walker, increasing convergence speed.
        seed: a seed for the random generator.
        verbose: print progress bar ? (default True)
        suppress_warnings: remove warnings ? (default False)
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.

    Returns:
        A Result object.
    """
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)

    params = initialize_sa(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability, epsilon,
                           T_0, tol, moves, bounds, fun, nb_cores, vectorized, vectorized_on_evaluations,
                           vectorized_skip_marker, backup, nb_slots, suppress_warnings, detect_convergence,
                           window_size, seed)

    x = params.base.x
    costs = params.costs
    last_ns = params.last_ns

    # initialize the trace history keeper
    trace = Trace(params.base.max_iter, x.shape, window_size=params.window_size,
                  detect_convergence=params.base.detect_convergence)
    trace.initialize(x, costs)

    if verbose:
        progress_bar = tqdm(range(params.base.max_iter), unit='iteration')
    else:
        progress_bar = range(params.base.max_iter)

    if params.parallel.vectorized:
        executor: Union[Type[VectorizedExecutor], Type[ParallelExecutor], Type[LinearExecutor]] = VectorizedExecutor
    elif params.parallel.nb_cores > 1:
        executor = ParallelExecutor
    else:
        executor = LinearExecutor

    # run the SA algorithm
    with executor(max_workers=params.parallel.nb_cores) as ex:
        try:
            for iteration in progress_bar:
                start = time.time()

                temperature = T(iteration, params.base.T_0, params.base.alpha)
                current_n = n(iteration, params.base.max_measures, params.base.sigma_max, params.base.T_0,
                              params.base.alpha, params.base.epsilon)
                accepted = [False for _ in range(params.parallel.nb_walkers)]
                rescued = [False for _ in range(params.parallel.nb_walkers)]

                acceptance_fraction = trace.mean_acceptance_fraction()

                updates = ex.map(params.fun,  # type: ignore
                                 x.copy(),
                                 costs,
                                 repeat(current_n),
                                 last_ns,
                                 repeat(params.base.args),
                                 repeat(params.moves.list_moves),
                                 repeat(params.moves.list_probabilities),
                                 repeat(iteration),
                                 repeat(params.base.max_iter),
                                 repeat(temperature),
                                 repeat(params.parallel.nb_slots_per_walker),
                                 repeat(acceptance_fraction if acceptance_fraction is not np.nan else 1.),
                                 positions=x.copy(),
                                 backup=params.backup,
                                 vectorized_on_evaluations=params.parallel.vectorized_on_evaluations,
                                 vectorized_skip_marker=params.parallel.vectorized_skip_marker)

                for _x, _cost, _last_n, _accepted, _walker_index in updates:
                    if _accepted:
                        x[_walker_index] = _x
                        costs[_walker_index] = _cost
                        last_ns[_walker_index] = _last_n
                        accepted[_walker_index] = _accepted

                index = trace.store(x, costs, temperature, current_n,
                                    sigma(iteration, params.base.T_0, params.base.alpha, params.base.epsilon), accepted)

                best_position, best_cost, best_index = trace.get_best()
                stuck_walkers = trace.are_stuck()

                for _walker_index in range(nb_walkers):
                    # rescue stuck walkers
                    if stuck_walkers[_walker_index]:
                        x[_walker_index] = best_position
                        costs[_walker_index] = best_cost
                        last_ns[_walker_index] = n(best_index[0], params.base.max_measures, params.base.sigma_max,
                                                   params.base.T_0, params.base.alpha, params.base.epsilon)
                        accepted[_walker_index] = True
                        rescued[_walker_index] = True

                trace.update(index, x, costs, rescued, best_position, best_cost, best_index, time.time() - start)

                if verbose:
                    progress_bar.set_description(f"T: "
                                                 f"{temperature:.4f}"
                                                 f"  A: "
                                                 f"{trace.mean_acceptance_fraction()*100:.4f}%"
                                                 f"  Best: "
                                                 f"{trace.get_best()[1]:.4f}"
                                                 f"  Current: "
                                                 f"{np.min(costs):.4f}")

                if trace.reached_convergence(params.base.tol):
                    message, success = 'Convergence tolerance reached.', True
                    break

            else:
                message, success = 'Requested number of iterations reached.', False

        except Exception as e:
            message, success = f'Unexpected failure : \n{e}', False

    trace.finalize()

    return Result(message, success, trace, params)


def restart(previous: Result,
            max_iter: int = 200,
            max_measures: int = 20,
            final_acceptance_probability: float = 1e-300,
            epsilon: float = 0.01,
            T_0: float = 5.,
            tol: float = 1e-3,
            verbose: bool = True,
            suppress_warnings: bool = False,
            detect_convergence: bool = True,
            window_size: Optional[int] = None) -> Result:
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
    # TODO
