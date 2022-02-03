# coding: utf-8
# Created on 03/02/2022 10:57
# Author : matteo

# ====================================================
# imports
import numbers
import collections.abc
import numpy as np
from warnings import warn
from multiprocessing import cpu_count
from dataclasses import dataclass, field

from typing import Sequence, Optional, Union, Callable, Any

from ..name_utils import ShapeError
from ..utils import get_slots_per_walker, get_evaluation_vectorized_mean_cost, get_walker_vectorized_mean_cost, \
    get_mean_cost
from ..moves import Move, SetStep, SetStretch, parse_moves
from ..__backup import Backup


# ====================================================
# code
@dataclass(frozen=True)
class BaseParameters:
    """
    Object for storing the general parameters used for running the SA algorithm.
    """
    args: tuple
    x0: np.ndarray
    max_iter: int
    max_measures: int
    final_acceptance_probability: float
    epsilon: float
    T_0: float
    tol: float
    alpha: float
    sigma_max: float
    suppress_warnings: bool
    detect_convergence: bool

    def __repr__(self) -> str:
        return f"\tx0: {self.x0}\n" \
               f"\tmax iterations: {self.max_iter}\n" \
               f"\tmax measures: {self.max_measures}\n" \
               f"\tfinal acceptance probability: {self.final_acceptance_probability}\n" \
               f"\tepsilon: {self.epsilon}\n" \
               f"\tT_0: {self.T_0}\n" \
               f"\ttolerance: {self.tol}\n" \
               f"\talpha: {self.alpha}\n" \
               f"\tmax sigma: {self.sigma_max}\n" \
               f"\twarnings: {not self.suppress_warnings}\n" \
               f"\tdetect convergence: {self.detect_convergence}\n" \
               f"\targs: {self.args}\n"

    @property
    def x(self) -> np.ndarray:
        return self.x0.copy()


@dataclass(frozen=True)
class ParallelParameters:
    """
    Object for storing the parameters managing the calls to parallel or vectorized cost functions.
    """
    nb_walkers: int
    nb_cores: int
    vectorized: bool
    vectorized_on_evaluations: bool
    vectorized_skip_marker: Any
    nb_slots_per_walker: list[int]

    def __repr__(self) -> str:
        return f"\tnb walkers: {self.nb_walkers}\n" \
               f"\tnb cores: {self.nb_cores}\n" \
               f"\tvectorized: {self.vectorized}\n" \
               f"\tvectorized on evaluations: {self.vectorized_on_evaluations}\n" \
               f"\tvectorized skip marker: {self.vectorized_skip_marker}\n" \
               f"\tnb slots per walker: {self.nb_slots_per_walker}\n"


@dataclass(frozen=True)
class MoveParameters:
    """
    Object for storing the moves and their probabilities.
    """
    list_probabilities: list[float]
    list_moves: list[Move]

    def __repr__(self) -> str:
        return '\t' + '\n\t'.join([f"{move.__class__.__name__} with probability {np.round(proba, 4)}"
                                   for move, proba in zip(self.list_moves, self.list_probabilities)]) + '\n'

    @property
    def using_SetMoves(self) -> bool:
        """
        Is at least one of the moves a Set move ?
        """
        return any([isinstance(move, (SetStep, SetStretch)) for move in self.list_moves])

    def set_bounds(self,
                   bounds: Union[tuple[float, float], Sequence[tuple[float, float]], None]) -> None:
        """
        Set the bounds for all moves.

        Args:
            bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
                (lower_bound, upper_bound)
                or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        """
        for move in self.list_moves:
            move.set_bounds(bounds)


@dataclass(frozen=True)
class SAParameters:
    """
    Object for storing the parameters used for running the SA algorithm.
    """
    base: BaseParameters
    parallel: ParallelParameters
    moves: MoveParameters
    fun: Callable[[np.ndarray, Any], Union[list[float], float]]
    backup: Backup = field(repr=False)
    costs: list[float]
    last_ns: list[int]
    window_size: int
    seed: int
    active_backup: bool = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'active_backup', self.backup.active)

    def __repr__(self) -> str:
        return f"{self.base}" \
               f"{self.parallel}\n" \
               f"Moves:\n{self.moves}\n" \
               f"Function: {self.fun.__name__}\n" \
               f"Window size: {self.window_size}\n" \
               f"Seed: {self.seed}\n" \
               f"Backup: {'active' if self.active_backup else 'no'}\n"  # type: ignore


def check_base_parameters(args: Optional[Sequence],
                          x0: np.ndarray,
                          nb_walkers: int,
                          max_iter: int,
                          max_measures: int,
                          final_acceptance_probability: float,
                          epsilon: float,
                          T_0: float,
                          tol: float,
                          suppress_warnings: bool,
                          detect_convergence: bool) -> BaseParameters:
    """
    Check validity of base parameters.

    Args:
        args: an optional sequence of arguments to pass to the function to minimize.
        x0: a <d> dimensional vector of initial values.
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)

    Returns:
        BaseParameters.
    """
    # arguments
    args = tuple(args) if args is not None else ()

    # initial values
    if x0.ndim == 1:
        if len(x0) > 1:
            x0 = np.array([x0 + np.random.uniform(-0.5e-10, 0.5e-10) for _ in range(nb_walkers)])

        else:
            x0 = np.array([x0])

    if x0.shape[0] != nb_walkers:
        raise ShapeError(f'Matrix of initial values should have {nb_walkers} rows (equal to the number of '
                         f'parallel walkers), not {x0.shape[0]}')

    if len(x0) > 1 and np.all([x0[0] == x0[i] for i in range(1, len(x0))]):
        warn('Initial positions are the same for all walkers, adding random noise.')

        x0 = np.array([x0[i] + np.random.uniform(-0.5e-10, 0.5e-10) for i in range(len(x0))])

    x0 = x0.astype(np.float32)

    # max iterations
    if max_iter < 0:
        raise ValueError("'max_iter' parameter must be positive.")

    max_iter = int(max_iter)

    # max function evaluations
    if max_measures < 0:
        raise ValueError("'max_measures' parameter must be positive.")

    max_measures = int(max_measures)

    # final acceptance probability
    if final_acceptance_probability < 0 or final_acceptance_probability > 1:
        raise ValueError(f"Invalid value '{final_acceptance_probability}' for 'final_acceptance_probability', "
                         f"should be in [0, 1].")

    # epsilon
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"Invalid value '{epsilon}' for 'epsilon', should be in (0, 1).")

    # T 0
    if T_0 < 0:
        raise ValueError("'T_0' parameter must be at least 0.")

    T_0 = float(T_0)

    # tolerance
    if tol <= 0:
        raise ValueError("'tol' parameter must be strictly positive.")

    # max sigma
    T_final = -1 / np.log(final_acceptance_probability)
    alpha = (T_final / T_0) ** (1 / max_iter)

    sigma_max = np.sqrt((max_measures - 1) * T_0 * alpha * (1 - epsilon)) / 3

    # suppress warnings
    if not suppress_warnings and max_iter < 200:
        warn('It is not recommended running the SA algorithm with less than 200 iterations.')

    return BaseParameters(args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol,
                          alpha, sigma_max, suppress_warnings, detect_convergence)


def check_parallel_parameters(nb_walkers: int,
                              nb_cores: int,
                              vectorized: bool,
                              vectorized_on_evaluations: bool,
                              vectorized_skip_marker: Any,
                              nb_slots: Optional[int]) -> ParallelParameters:
    """
    Check validity of parallel parameters.

    Args:
        nb_walkers: the number of parallel walkers in the ensemble.
        nb_cores: number of cores that can be used to move walkers in parallel.
        vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
            just one. (<nb_cores> parameter will be set to 1 in this case.)
        vectorized_on_evaluations: vectorize <fun> calls on evaluations (or walkers) ?
        vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
            evaluation for a particular position vector can be skipped.
        nb_slots: When using a vectorized function, the total number of position vectors for which the cost can be
            computed at once.

    Returns:
        ParallelParameters.
    """
    # nb_cores
    nb_cores = int(nb_cores)

    if vectorized:
        nb_cores = 1

    if nb_cores < 1:
        raise ValueError('Cannot use less than one core.')

    if nb_cores > cpu_count():
        raise ValueError(f"Cannot use more than available CPUs ({cpu_count()}).")

    # init nb_slots per walker
    if nb_slots is None:
        nb_slots_per_walker = [1 for _ in range(nb_walkers)]

    elif not vectorized:
        raise ValueError("Cannot use slots unless using a vectorized cost function.")

    elif nb_slots < nb_walkers:
        raise ValueError(f"nb_slots ({nb_slots}) is less than the number of walkers ({nb_walkers})!")

    else:
        nb_slots_per_walker = get_slots_per_walker(nb_slots, nb_walkers)

    return ParallelParameters(nb_walkers, nb_cores, vectorized, vectorized_on_evaluations, vectorized_skip_marker,
                              nb_slots_per_walker)


def check_bounds(bounds: Optional[Union[tuple[float, float], Sequence[tuple[float, float]]]],
                 x0: np.ndarray) -> None:
    """
    Check validity of bounds.

    Args:
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        x0: a <d> dimensional vector of initial values or a matrix of initial values of shape (nb_walkers, d).
    """
    if bounds is not None:
        if isinstance(bounds, tuple) and isinstance(bounds[0], numbers.Number) \
                and isinstance(bounds[1], numbers.Number):
            if np.any(x0 < bounds[0]) or np.any(x0 > bounds[1]):
                raise ValueError('Some values in x0 do not lie in between defined bounds.')

        elif isinstance(bounds, collections.abc.Sequence):
            if len(bounds) != x0.shape[1]:
                raise ShapeError(f'Bounds must be defined for all dimensions, but only {len(bounds)} out of'
                                 f' {x0.shape[1]} were defined.')

            for dim_index, bound in enumerate(bounds):
                if isinstance(bound, tuple) and isinstance(bound[0], numbers.Number) and isinstance(bound[1],
                                                                                                    numbers.Number):
                    if np.any(x0[:, dim_index] < bound[0]) or np.any(x0[:, dim_index] > bound[1]):
                        raise ValueError(f'Some values in x0 do not lie in between defined bounds for dimensions '
                                         f'{dim_index}.')

                else:
                    raise TypeError(
                        "'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                        "with the following format: \n"
                        "\t(lower_bound, upper_bound)\n "
                        "or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.")

        else:
            raise TypeError(
                "'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                "with the following format: \n"
                "\t(lower_bound, upper_bound)\n "
                "or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.")


def initialize_sa(args: Optional[Sequence],
                  x0: np.ndarray,
                  nb_walkers: int,
                  max_iter: int,
                  max_measures: int,
                  final_acceptance_probability: float,
                  epsilon: float,
                  T_0: float,
                  tol: float,
                  moves: Union[Move, Sequence[Move], Sequence[tuple[float, Move]]],
                  bounds: Optional[Union[tuple[float, float], Sequence[tuple[float, float]]]],
                  fun: Callable[[np.ndarray, Any], Union[list[float], float]],
                  nb_cores: int,
                  vectorized: bool,
                  vectorized_on_evaluations: bool,
                  vectorized_skip_marker: Any,
                  backup: bool,
                  nb_slots: Optional[int],
                  suppress_warnings: bool,
                  detect_convergence: bool,
                  window_size: Optional[int],
                  seed: int) -> SAParameters:
    """
    Check validity of parameters and compute initial values before running the SA algorithm.

    Args:
        args: an optional sequence of arguments to pass to the function to minimize.
        x0: a <d> dimensional vector of initial values or a matrix of initial values of shape (nb_walkers, d).
        nb_walkers: the number of parallel walkers in the ensemble.
        max_iter: the maximum number of iterations before stopping the algorithm.
        max_measures: the maximum number of function evaluations to average per step.
        final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
            steeper descent profiles)
        T_0: initial temperature value.
        tol: the convergence tolerance.
        moves: either
                    - a single josiann.Move object
                    - a sequence of josiann.Move objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.Move)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
        bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
            (lower_bound, upper_bound)
            or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
        fun: a <d> dimensional (noisy) function to minimize.
        nb_cores: number of cores that can be used to move walkers in parallel.
        vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
            just one. (<nb_cores> parameter will be set to 1 in this case.)
        vectorized_on_evaluations: vectorize <fun> calls on evaluations (or walkers) ?
        vectorized_skip_marker: when vectorizing on walkers, the object to pass to <fun> to indicate that an
            evaluation for a particular position vector can be skipped.
        backup: use Backup for storing previously computed function evaluations and reusing them when returning to
            the same position vector ? (Only available when using SetStep moves).
        nb_slots: When using a vectorized function, the total number of position vectors for which the cost can be
            computed at once.
        suppress_warnings: remove warnings ?
        detect_convergence: run convergence detection for an early stop of the algorithm ? (default True)
        window_size: number of past iterations to look at for detecting the convergence, getting the best position
            and computing the acceptance fraction.
        seed: a seed for the random generator.

    Returns:
        Valid parameters and initial values.
    """
    # base parameters
    base_parameters = check_base_parameters(args, x0, nb_walkers, max_iter, max_measures, final_acceptance_probability,
                                            epsilon, T_0, tol, suppress_warnings, detect_convergence)

    # parallel parameters
    parallel_parameters = check_parallel_parameters(nb_walkers, nb_cores, vectorized, vectorized_on_evaluations,
                                                    vectorized_skip_marker, nb_slots)

    # bounds
    check_bounds(bounds, base_parameters.x0)

    # move parameters
    move_parameters = MoveParameters(*parse_moves(moves, nb_walkers))
    move_parameters.set_bounds(bounds)

    # init backup storage
    backup_storage = Backup(active=move_parameters.using_SetMoves and backup)

    # initial costs and last_ns
    if vectorized:
        if vectorized_on_evaluations:
            costs = get_evaluation_vectorized_mean_cost(
                fun,
                base_parameters.x0,
                1,
                base_parameters.args,
                [(0, 0.) for _ in range(len(base_parameters.x0))]
            )

        else:
            init_x = np.zeros((sum(parallel_parameters.nb_slots_per_walker), x0.shape[1]))
            init_x[0:len(base_parameters.x0)] = base_parameters.x0
            costs = get_walker_vectorized_mean_cost(
                fun,
                init_x,
                1,
                base_parameters.args,
                [(0, 0.) for _ in range(len(base_parameters.x0))] +
                [(max_iter, 0.) for _ in range(sum(parallel_parameters.nb_slots_per_walker) - len(base_parameters.x0))],
                vectorized_skip_marker
            )[0:len(base_parameters.x0)]

    else:
        costs = [get_mean_cost(fun, x_vector, 1, base_parameters.args, (0, 0.)) for x_vector in base_parameters.x0]

    last_ns = [1 for _ in range(nb_walkers)]

    # window size
    if window_size is not None:
        if max_iter < window_size < 1:
            raise ValueError(f"Invalid window size '{window_size}', should be in [{1}, {max_iter}].")

    else:
        # window_size = max(1, min(50, int(0.1 * max_iter)))
        window_size = max(50, int(0.1 * max_iter))

    return SAParameters(base_parameters, parallel_parameters, move_parameters, fun, backup_storage, costs, last_ns,
                        window_size, seed)
