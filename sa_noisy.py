# coding: utf-8
# Created on 23/07/2021 23:28
# Author : matteo

# ====================================================
# imports
import warnings
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from dataclasses import dataclass, field
from plotly.subplots import make_subplots

from typing import Callable, Tuple, Optional, Sequence, Union, Any

from .utils import parse_moves, get_mean_cost
from .name_utils import ShapeError
from .moves import Move, RandomStep


# ====================================================
# code
class Trace:
    """
    Object for storing the trace history of an SA run.
    """

    def __init__(self, nb_iterations: int, ndim: int, window_size: int):
        """
        :param nb_iterations: number of expected iterations for the SA algorithm.
        :param ndim: number of dimensions of the vector to optimize.
        :param window_size: size of the window of values to test for convergence.
        """
        self.__position_trace = np.zeros((nb_iterations, ndim))
        self.__cost_trace = np.zeros(nb_iterations)
        self.__temperature_trace = np.zeros(nb_iterations)
        self.__n_trace = np.zeros(nb_iterations)
        self.__sigma_trace = np.zeros(nb_iterations)

        self.__initialized = False

        self.__window_size = window_size
        self.__iteration_counter = 0

    def initialize(self, position: np.ndarray, cost: float) -> None:
        """
        Save state zero before running the SA algorithm. This function should be called before actually storing run
            values.

        :param position: the initial vector.
        :param cost: the initial cost.
        """
        self.__initialized = True

        np.insert(self.__position_trace, 0, position.copy())
        np.insert(self.__cost_trace, 0, float(cost))

    def finalize(self) -> None:
        """
        Cleanup traces at the end of the SA algorithm.
        """
        self.__position_trace = self.__position_trace[:self.__iteration_counter]
        self.__cost_trace = self.__cost_trace[:self.__iteration_counter]
        self.__temperature_trace = self.__temperature_trace[:self.__iteration_counter]
        self.__n_trace = self.__n_trace[:self.__iteration_counter]
        self.__sigma_trace = self.__sigma_trace[:self.__iteration_counter]

    def __check_initialized(self) -> None:
        """
        Check this Trace has been correctly initialized.
        """
        if not self.__initialized:
            warnings.warn('Trace was not initialized, skipping initial state.', RuntimeWarning)

            self.__initialized = True

    def store(self, position: np.ndarray, cost: float, temperature: float, n: int, sigma: float) -> None:
        """
        Save the current position of the vector to optimize, the current cost, temperature and number of averaged
            function evaluations.

        :param position: the current vector.
        :param cost: the current cost.
        :param temperature: the current temperature.
        :param n: the current number of averaged function evaluations.
        :param sigma: the current estimated standard deviation.
        """
        self.__check_initialized()

        self.__position_trace[self.__iteration_counter] = position.copy()
        self.__cost_trace[self.__iteration_counter] = float(cost)
        self.__temperature_trace[self.__iteration_counter] = float(temperature)
        self.__n_trace[self.__iteration_counter] = int(n)
        self.__sigma_trace[self.__iteration_counter] = float(sigma)

        self.__iteration_counter += 1

    def reached_convergence(self, tolerance: float) -> bool:
        """
        Has the cost trace reached convergence within a tolerance margin ?

        :param tolerance: the allowed difference between the last 2 costs.

        :return: Whether the cost trace has converged.
        """
        if self.__iteration_counter < self.__window_size:
            return False

        mean_window = np.mean(self.__cost_trace[self.__iteration_counter-self.__window_size:self.__iteration_counter])
        RMSD = np.sqrt(np.sum(
            (self.__cost_trace[self.__iteration_counter-self.__window_size:self.__iteration_counter] - mean_window) ** 2
        ) / (self.__window_size - 1))

        return RMSD < tolerance

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the vector the optimize.

        :return: The number of dimensions of the vector the optimize.
        """
        return self.__position_trace.shape[1]

    @property
    def nb_positions(self) -> int:
        return self.__position_trace.shape[0]

    @property
    def nb_iterations(self) -> int:
        return self.__cost_trace.shape[0]

    def get_best(self) -> Tuple[np.ndarray, float, int]:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        :return: the best vector, best cost and iteration number that reached it.
        """
        _best_index = np.argmin(
            self.__cost_trace[self.__iteration_counter - self.__window_size:self.__iteration_counter]) + \
            self.__iteration_counter - self.__window_size

        return self.__position_trace[_best_index], self.__cost_trace[_best_index], _best_index

    def get_position_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get traces for vector and cost values along iterations.

        :return: Traces for vector and cost values along iterations.
        """
        return self.__position_trace, self.__cost_trace

    def get_parameters_trace(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get traces related to parameters values along iterations.

        :return: Traces related to parameters values along iterations.
        """
        return self.__temperature_trace, self.__n_trace, self.__sigma_trace

    def plot_positions(self, true_values: Optional[Sequence[float]] = None) -> None:
        """
        Plot reached positions and costs for the vector to optimize along iterations.

        :param true_values: an optional sequence of known true values for each dimension of the vector to optimize.
        """
        if true_values is not None and len(true_values) != self.ndim:
            raise ShapeError(f'The vector of true values should have {self.ndim} dimensions, not {len(true_values)}.')

        fig = make_subplots(rows=self.ndim + 1, cols=1, shared_xaxes=True)

        fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                 y=self.__cost_trace,
                                 name='cost',
                                 marker=dict(color='rgba(0, 0, 200, 1)'),
                                 hovertext=self.__cost_trace), row=1, col=1)

        for i in range(self.ndim):
            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=self.__position_trace[:, i],
                                     marker=dict(color='rgba(0, 0, 0, 1)'),
                                     name=f'Dimension #{i}'), row=i+2, col=1)
            if true_values is not None:
                fig.add_trace(go.Scatter(x=[0, self.nb_positions],
                                         y=[true_values[i], true_values[i]],
                                         mode='lines',
                                         marker=dict(color='rgba(200, 0, 0, 0.8)'),
                                         showlegend=False), row=i+2, col=1)

        fig.show()

    def plot_parameters(self) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.
        """
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__temperature_trace,
                                 name='T',
                                 hovertext=self.__temperature_trace), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__n_trace,
                                 name='n',
                                 hovertext=self.__n_trace), row=3, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__sigma_trace,
                                 name='sigma',
                                 hovertext=self.__sigma_trace), row=2, col=1)

        fig.show()


@dataclass
class Result:
    """
    Object for storing the results of a run.
    """
    message: str
    success: bool
    trace: Trace
    args: Sequence
    x0: np.ndarray
    max_iter: int
    max_measures: int
    final_acceptance_probability: float
    epsilon: float
    T_0: float
    tol: float
    window_size: int
    alpha: float
    T_final: float
    computed_T_0: bool
    x: np.ndarray = field(init=False)
    x_cost: float = field(init=False)
    x_iter: int = field(init=False)

    def __post_init__(self):
        self.x, self.x_cost, self.x_iter = self.trace.get_best()

    def __repr__(self) -> str:
        T_0_string = f"\tT_0 : {self.T_0}\n"

        return f"Message : {self.message}\n" \
               f"User parameters : \n" \
               f"\targs : {self.args}\n" \
               f"\tx0 : {self.x0}\n"\
               f"\tmax_iter : {self.max_iter}\n" \
               f"\tmax_measures : {self.max_measures}\n" \
               f"\tfinal_acceptance_probability : {self.final_acceptance_probability}\n" \
               f"\tepsilon : {self.epsilon}\n" \
               f"{T_0_string if not self.computed_T_0 else ''}" \
               f"\ttol : {self.tol}\n"\
               f"\twindow_size : {self.window_size}\n" \
               f"Computed parameters : \n" \
               f"{T_0_string if self.computed_T_0 else ''}" \
               f"\talpha : {self.alpha}\n" \
               f"\tT_final : {self.T_final}\n" \
               f"Success : {self.success}\n" \
               f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter})\n" \
               f"x: {self.x}"
    

@dataclass
class Sigma:
    T_0: float
    alpha: float
    epsilon: float
    __factor: float = field(init=False)
    
    def __post_init__(self):
        self.__factor = self.alpha * (1 - self.epsilon)
    
    def get(self, iteration) -> float:
        return self.T_0 * self.__factor ** iteration


def acceptance_probability(current_cost: float, new_cost: float, T: float) -> float:
    """
    Compute the acceptance probability for a new proposed cost, given the current cost and a temperature.

    :param current_cost: the current cost.
    :param new_cost: the new proposed cost.
    :param T: the current temperature.

    :return: the probability of acceptance of the new proposed cost.
    """
    return (current_cost - new_cost) / T


def get_delta_max() -> float:
    """TODO"""
    raise NotImplementedError


def sa(fun: Callable[[np.ndarray, Any], float],
       x0: np.ndarray,
       args: Optional[Sequence] = None,
       bounds: Optional[Sequence[Tuple[float, float]]] = None,
       moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]] = ((0.8, RandomStep(0.05)),
                                                                           (0.2, RandomStep(0.5))),
       max_iter: int = 200,
       max_measures: int = 20,
       final_acceptance_probability: float = 1e-5,
       epsilon: float = 0.01,
       T_0: Optional[float] = None,
       tol: float = 1e-3,
       window_size: int = 100) -> Result:
    """
    Simulated Annealing for minimizing noisy cost functions.

    :param fun: a <d> dimensional (noisy) function to minimize.
    :param x0: a <d> dimensional vector of initial values.
    :param args: an optional sequence of arguments to pass to the function to minimize.
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
    :param moves: either
                    - a single josiann.Move object
                    - a sequence of josiann.Move objects (all Moves have the same probability of being selected at
                        each step for proposing a new candidate vector x)
                    - a sequence of tuples with the following format :
                        (selection probability, josiann.Move)
                        In this case, the selection probability dictates the probability of each Move of being
                        selected at each step.
    :param max_iter: the maximum number of iterations before stopping the algorithm.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)
    :param T_0: optional initial temperature value.
    :param tol: the convergence tolerance.
    :param window_size: a window of the last <window_size> cost values are used to test for convergence.

    :return: a Result object.
    """
    def check_parameters(_args, _x0, _max_iter, _max_measures, _final_acceptance_probability,
                         _epsilon, _T_0, _tol, _window_size):
        _args = _args if _args is not None else ()

        if _x0.ndim != 1:
            raise ShapeError(f'Vector of initial values should be one dimensional, not {x0.ndim}.')

        if _max_iter < 0:
            raise ValueError("'max_iter' parameter must be positive.")
        else:
            _max_iter = int(_max_iter)

        if _max_measures < 0:
            raise ValueError("'max_measures' parameter must be positive.")
        else:
            _max_measures = int(_max_measures)

        if _final_acceptance_probability < 0 or _final_acceptance_probability > 1:
            raise ValueError(f"Invalid value '{_final_acceptance_probability}' for 'final_acceptance_probability', "
                             f"should be in [0, 1].")

        if _epsilon <= 0 or _epsilon >= 1:
            raise ValueError(f"Invalid value '{_epsilon}' for 'epsilon', should be in (0, 1).")

        if _T_0 is not None and _T_0 < 0:
            raise ValueError("'T_0' parameter must be at least 0.")

        if _T_0 is None:
            _T_0 = -get_delta_max() / np.log(0.8)
            _computed_T_0 = True
        else:
            _T_0 = float(_T_0)
            _computed_T_0 = False

        if _tol <= 0:
            raise ValueError("'tol' parameter must be strictly positive.")

        if _window_size < 1:
            raise ValueError("'window_size' parameter must be greater than 0.")
        else:
            _window_size = int(_window_size)

        return _args, _x0, _max_iter, _max_measures, _final_acceptance_probability, _epsilon, \
            _T_0, _tol, _window_size, _computed_T_0

    def sigma(k: int) -> float:
        """
        Compute the estimated standard deviation at iteration k.

        :param k: the iteration number.

        :return: the estimated standard deviation.
        """
        return T_0 * (alpha * (1 - epsilon)) ** k

    def n(k: int) -> int:
        """
        Compute the number of necessary measures at iteration k.

        :param k: the iteration number.

        :return: the number of necessary measures.
        """
        return int(np.ceil((max_measures * sigma_max ** 2) / ((max_measures - 1) * sigma(k) ** 2 + sigma_max ** 2)))

    def T(k: int) -> float:
        """
        Compute the temperature at iteration k.

        :param k: the iteration number.

        :return: the temperature.
        """
        return T_0 * alpha ** k

    # check parameters
    args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, \
        computed_T_0 = check_parameters(args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0,
                                        tol, window_size)

    # get moves and associated probabilities
    list_probabilities, list_moves = parse_moves(moves)

    for move in list_moves:
        move.set_bounds(bounds)

    # initial state
    x = x0.copy()
    cost = get_mean_cost(fun, x, 1, *args)

    # initialize Temperature
    T_final = -1 / np.log(final_acceptance_probability)
    
    # initialize parameters
    alpha = (T_final / T_0) ** (1 / max_iter)
    sigma_max = T_0

    # initialize the trace history keeper
    trace = Trace(max_iter, len(x0), window_size=window_size)
    trace.initialize(x, cost * n(0) / n(1))

    # run the SA algorithm
    for iteration in tqdm(range(max_iter), unit='iteration'):

        temperature = T(iteration)
        nb_measures = n(iteration)

        # pick a move at random from available moves
        move = np.random.choice(list_moves, p=list_probabilities)

        # generate a new proposal as a neighbor of x and get its cost
        proposed_x = move.get_proposal(x)
        proposed_cost = get_mean_cost(fun, proposed_x, nb_measures, *args)
        normalized_cost = cost * n(iteration - 1) / nb_measures

        if acceptance_probability(normalized_cost, proposed_cost, temperature) > np.log(np.random.random()):
            x, cost = proposed_x, proposed_cost

        trace.store(x, cost * nb_measures / n(iteration + 1), temperature, nb_measures, sigma(iteration))

        if trace.reached_convergence(tol):
            message = 'Convergence tolerance reached.'
            break

    else:
        message = 'Requested number of iterations reached.'

    trace.finalize()

    return Result(message, True, trace, args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0,
                  tol, window_size, alpha, T_final, computed_T_0)
