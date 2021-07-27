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

    def __init__(self, nb_iterations: int, nb_repeats: int, ndim: int, window_size: int):
        """
        :param nb_iterations: number of expected iterations for the SA algorithm.
        :param nb_repeats: number of expected repeats for each iteration.
        :param ndim: number of dimensions of the vector to optimize.
        :param window_size: size of the window of values to test for convergence.
        """
        self.__position_trace = np.zeros((nb_iterations * nb_repeats + 1, ndim))
        self.__cost_trace = np.zeros(nb_iterations * nb_repeats + 1)
        self.__temperature_trace = np.zeros(nb_iterations)
        self.__nb_repeats_trace = np.zeros(nb_iterations)
        self.__n_trace = np.zeros(nb_iterations)

        self.__initialized = False

        self.__window_size = window_size
        self.__position_counter = 0
        self.__iteration_counter = 0

    def initialize(self, position: np.ndarray, cost: float) -> None:
        """
        Save state zero before running the SA algorithm. This function should be called before actually storing run
            values.

        :param position: the initial vector.
        :param cost: the initial cost.
        """
        self.__initialized = True

        self.store_position(position, cost)

    def finalize(self) -> None:
        """
        Cleanup traces at the end of the SA algorithm.
        """
        self.__position_trace = self.__position_trace[:self.__position_counter]
        self.__cost_trace = self.__cost_trace[:self.__position_counter]
        self.__temperature_trace = self.__temperature_trace[:self.__iteration_counter]
        self.__nb_repeats_trace = self.__nb_repeats_trace[:self.__iteration_counter]
        self.__n_trace = self.__n_trace[:self.__iteration_counter]

    def __check_initialized(self) -> None:
        """
        Check this Trace has been correctly initialized.
        """
        if not self.__initialized:
            warnings.warn('Trace was not initialized, skipping initial state.', RuntimeWarning)

            self.__position_trace = self.__position_trace[1:]
            self.__cost_trace = self.__cost_trace[1:]

            self.__initialized = True

    def store_position(self, position: np.ndarray, cost: float) -> None:
        """
        Save the current position of the vector to optimize (at each repeat in each iteration).

        :param position: the current vector.
        :param cost: the current cost.
        """
        self.__check_initialized()

        self.__position_trace[self.__position_counter] = position.copy()
        self.__cost_trace[self.__position_counter] = float(cost)

        self.__position_counter += 1

    def store_iteration(self, temperature: float, nb_repeats: int, n: int) -> None:
        """
        Save the current cost, temperature and number of averaged function evaluations.

        :param temperature: the current temperature.
        :param nb_repeats: the current number of repeats in an iteration.
        :param n: the current number of averaged function evaluations.
        """
        self.__check_initialized()

        self.__temperature_trace[self.__iteration_counter] = float(temperature)
        self.__nb_repeats_trace[self.__iteration_counter] = int(nb_repeats)
        self.__n_trace[self.__iteration_counter] = int(n)

        self.__iteration_counter += 1

    def reached_convergence(self, tolerance: float) -> bool:
        """
        Has the cost trace reached convergence within a tolerance margin ?

        :param tolerance: the allowed difference between the last 2 costs.

        :return: Whether the cost trace has converged.
        """
        if self.__position_counter < self.__window_size:
            return False

        mean_window = np.mean(self.__cost_trace[self.__position_counter-self.__window_size:self.__position_counter])
        RMSD = np.sqrt(np.sum(
            (self.__cost_trace[self.__position_counter-self.__window_size:self.__position_counter] - mean_window) ** 2
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
            self.__cost_trace[self.__position_counter - self.__window_size:self.__position_counter]) + \
            self.__position_counter - self.__window_size

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
        return self.__temperature_trace, self.__nb_repeats_trace, self.__n_trace

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
                                 y=self.__nb_repeats_trace,
                                 name='repeats',
                                 hovertext=self.__nb_repeats_trace), row=2, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__n_trace,
                                 name='n',
                                 hovertext=self.__n_trace), row=3, col=1)

        fig.show()


@dataclass
class Result:
    """
    Object for storing the results of a run.
    """
    message: str
    success: bool
    trace: Trace
    x: np.ndarray = field(init=False)
    x_cost: float = field(init=False)
    x_iter: int = field(init=False)

    def __post_init__(self):
        self.x, self.x_cost, self.x_iter = self.trace.get_best()

    def __repr__(self) -> str:
        return f"Message : {self.message}\n" \
               f"Success : {self.success}\n" \
               f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter})\n" \
               f"x: {self.x}"


def acceptance_probability(current_cost: float, new_cost: float, T: float) -> float:
    """
    Compute the acceptance probability for a new proposed cost, given the current cost and a temperature.

    :param current_cost: the current cost.
    :param new_cost: the new proposed cost.
    :param T: the current temperature.

    :return: the probability of acceptance of the new proposed cost.
    """
    return (current_cost - new_cost) / T


def sa(fun: Callable[[np.ndarray, Any], float],
       x0: np.ndarray,
       args: Optional[Sequence] = None,
       bounds: Optional[Sequence[Tuple[float, float]]] = None,
       moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]] = ((0.8, RandomStep(0.05)),
                                                                           (0.2, RandomStep(0.5))),
       max_iter: int = 200,
       max_repeats: int = 50,
       min_repeats: int = 10,
       T_max: float = 1e5,
       alpha: float = 0.9,
       beta: float = 0.1,
       increase_interval: float = 1.0,
       tol: float = 1e-3,
       window_size: int = 100) -> Result:
    """
    Simulated Annealing for minimizing noisy cost functions.

    :param fun: a <n> dimensional (noisy) function to minimize.
    :param x0: a <n> dimensional vector of initial values.
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
    :param max_repeats: the maximum number of repeats to perform per iteration.
    :param min_repeats: the minimum number of repeats to perform per iteration.
    :param T_max: The initial maximal temperature.
    :param alpha: The rate of temperature decrease.
    :param beta: Parameter for modulating the rate of decrease of repeats per iteration.
    :param increase_interval: the rate of increase of evaluations of the function.
    :param tol: the convergence tolerance.
    :param window_size: a window of the last <window_size> cost values are used to test for convergence.

    :return: a Result object.
    """
    args = args if args is not None else ()

    if x0.ndim != 1:
        raise ShapeError(f'Vector of initial values should be one dimensional, not {x0.ndim}.')

    # get moves and associated probabilities
    list_probabilities, list_moves = parse_moves(moves)

    for move in list_moves:
        move.set_bounds(bounds)

    # initial state
    x = x0.copy()
    cost = get_mean_cost(fun, x, 1, *args)

    # initialize the trace history keeper
    trace = Trace(max_iter, max_repeats, len(x0), window_size=window_size)
    trace.initialize(x, cost)

    # run the SA algorithm
    for iteration in tqdm(range(max_iter), unit='iteration'):

        T = T_max * alpha ** iteration
        nb_repeats = int(np.floor((max_repeats - min_repeats) * alpha ** (iteration * beta) + min_repeats))
        n = max(int(np.floor(iteration ** 2 / increase_interval)), 1)

        for repeat in range(nb_repeats):
            # pick a move at random from available moves
            move = np.random.choice(list_moves, p=list_probabilities)

            # generate a new proposal as a neighbor of x and get its cost
            proposed_x = move.get_proposal(x)
            proposed_cost = get_mean_cost(fun, proposed_x, n, *args)

            if acceptance_probability(cost, proposed_cost, T) > np.log(np.random.random()):
                x, cost = proposed_x, proposed_cost

            trace.store_position(x, cost)

        trace.store_iteration(T, nb_repeats, n)

        if trace.reached_convergence(tol):
            message = 'Convergence tolerance reached.'
            break

    else:
        message = 'Requested number of iterations reached.'

    trace.finalize()

    return Result(message, True, trace)
