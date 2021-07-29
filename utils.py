# coding: utf-8
# Created on 26/07/2021 12:08
# Author : matteo

# ====================================================
# imports
import warnings
import collections
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field
from plotly.subplots import make_subplots

from typing import Union, Sequence, Tuple, List, Callable, Any, Optional
from typing_extensions import Literal

from .name_utils import ShapeError
from .__moves import Move


# ====================================================
# code
class Trace:
    """
    Object for storing the trace history of an SA run.
    """

    def __init__(self,
                 nb_iterations: int,
                 shape: Tuple[int, int],
                 window_size: int):
        """
        :param nb_iterations: number of expected iterations for the SA algorithm.
        :param shape: shape of the matrices to store (nb_walkers, d).
        :param window_size: size of the window of values to test for convergence.
        """
        self.__position_trace = np.zeros((nb_iterations, shape[0], shape[1]))
        self.__cost_trace = np.zeros(nb_iterations, shape[0])
        self.__temperature_trace = np.zeros(nb_iterations)
        self.__n_trace = np.zeros(nb_iterations)
        self.__sigma_trace = np.zeros(nb_iterations)
        self.__accepted = np.zeros((nb_iterations, shape[0]), dtype=bool)

        self.__initialized = False

        self.__window_size = window_size
        self.__iteration_counter = 0

    def initialize(self,
                   position: np.ndarray,
                   costs: List[float]) -> None:
        """
        Save state zero before running the SA algorithm. This function should be called before actually storing run
            values.

        :param position: the initial vector.
        :param costs: the initial costs.
        """
        self.__initialized = True

        np.insert(self.__position_trace, 0, position.copy())
        np.insert(self.__cost_trace, 0, np.array(costs))

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

    def store(self,
              position: np.ndarray,
              costs: List[float],
              temperature: float,
              _n: int,
              _sigma: float,
              accepted: List[bool]) -> None:
        """
        Save the current position of the vector to optimize, the current cost, temperature and number of averaged
            function evaluations.

        :param position: the current vector.
        :param costs: the current costs.
        :param temperature: the current temperature.
        :param _n: the current number of averaged function evaluations.
        :param _sigma: the current estimated standard deviation.
        :param accepted: were the current propositions accepted ?
        """
        self.__check_initialized()

        self.__position_trace[self.__iteration_counter] = position.copy()
        self.__cost_trace[self.__iteration_counter] = np.array(costs)
        self.__temperature_trace[self.__iteration_counter] = float(temperature)
        self.__n_trace[self.__iteration_counter] = int(_n)
        self.__sigma_trace[self.__iteration_counter] = float(_sigma)
        self.__accepted[self.__iteration_counter] = np.array(accepted)

        self.__iteration_counter += 1

    def reached_convergence(self,
                            tolerance: float) -> bool:
        """
        Has the cost trace reached convergence within a tolerance margin ?

        :param tolerance: the allowed difference between the last 2 costs.

        :return: Whether the cost trace has converged.
        """
        # TODO : redo with parallel workers
        return False

        if self.__iteration_counter < self.__window_size:
            return False

        mean_window = np.mean(self.__cost_trace[self.__iteration_counter - self.__window_size:self.__iteration_counter])
        RMSD = np.sqrt(np.sum(
            (self.__cost_trace[
             self.__iteration_counter - self.__window_size:self.__iteration_counter] - mean_window) ** 2
        ) / (self.__window_size - 1))

        return RMSD < tolerance

    def acceptance_fraction(self) -> List[float]:
        """
        Get the proportion of accepted proposition in the last <window_size> propositions.

        :return: The proportion of accepted proposition in the last <window_size> propositions.
        """
        if self.__iteration_counter < self.__window_size:
            return np.nan

        return [np.sum(self.__accepted[w, self.__iteration_counter - self.__window_size:self.__iteration_counter]) /
                self.__window_size for w in range(self.nb_walkers)]

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the vector the optimize.

        :return: The number of dimensions of the vector the optimize.
        """
        return self.__position_trace.shape[2]

    @property
    def nb_walkers(self) -> int:
        """
        Get the number of parallel walkers.

        :return: The number of parallel walkers.
        """
        return self.__position_trace.shape[1]

    @property
    def nb_positions(self) -> int:
        return self.__position_trace.shape[0]

    @property
    def nb_iterations(self) -> int:
        return self.__cost_trace.shape[0]

    def get_best(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        :return: the best vector, best cost and iteration number that reached it.
        """
        _best_index = np.argmin(
            self.__cost_trace[max(0, self.__iteration_counter - self.__window_size):self.__iteration_counter], axis=1)

        _best_index[0] += self.__iteration_counter - self.__window_size

        return self.__position_trace[_best_index[0], _best_index[1]], \
            self.__cost_trace[_best_index[0], _best_index[1]], \
            _best_index

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

        for w in range(self.nb_walkers):
            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=self.__cost_trace[:, w],
                                     name=f'costs[{w}]',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=self.__cost_trace), row=1, col=1)

        for i in range(self.ndim):
            for w in range(self.nb_walkers):
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self.__position_trace[:, w, i],
                                         marker=dict(color='rgba(0, 0, 0, 0.3)'),
                                         name=f'Dimension[{w}] #{i}'), row=i + 2, col=1)

            if true_values is not None:
                fig.add_trace(go.Scatter(x=[0, self.nb_positions],
                                         y=[true_values[i], true_values[i]],
                                         mode='lines',
                                         marker=dict(color='rgba(200, 0, 0, 1)'),
                                         showlegend=False), row=i + 2, col=1)

        fig.show()

    def plot_parameters(self) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.
        """
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__temperature_trace,
                                 name='T',
                                 hovertext=self.__temperature_trace), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__sigma_trace,
                                 name='sigma',
                                 hovertext=self.__sigma_trace), row=2, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__n_trace,
                                 name='n',
                                 hovertext=self.__n_trace), row=3, col=1)

        for w in range(self.nb_walkers):
            accepted_proportion = np.concatenate((np.array([np.nan for _ in range(self.__window_size)]),
                                                  np.convolve(self.__accepted[:, w],
                                                              np.ones(self.__window_size) / self.__window_size,
                                                              mode='valid') * 100))

            fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                     y=accepted_proportion,
                                     name=f'% accepted[{w}]',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=accepted_proportion), row=4, col=1)

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
               f"\tx0 : {self.x0}\n" \
               f"\tmax_iter : {self.max_iter}\n" \
               f"\tmax_measures : {self.max_measures}\n" \
               f"\tfinal_acceptance_probability : {self.final_acceptance_probability}\n" \
               f"\tepsilon : {self.epsilon}\n" \
               f"{T_0_string if not self.computed_T_0 else ''}" \
               f"\ttol : {self.tol}\n" \
               f"\twindow_size : {self.window_size}\n" \
               f"Computed parameters : \n" \
               f"{T_0_string if self.computed_T_0 else ''}" \
               f"\talpha : {self.alpha}\n" \
               f"\tT_final : {self.T_final}\n" \
               f"Success : {self.success}\n" \
               f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter})\n" \
               f"x: {self.x}"


def parse_moves(moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]],
                mtype: Literal['SingleMove', 'EnsembleMove']) -> Tuple[List[float], List[Move]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    :param moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
        sequence of tuples with format (probability: float, Move).
    :param mtype: the type of moves that are accepted (either 'SingleMove' or 'EnsembleMove').

    :return: the list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.Sequence) or isinstance(moves, str):
        if isinstance(moves, Move) and moves.__class__.__name__ == mtype:
            return [1.0], [moves]

        raise ValueError(f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a "
                         f"'{mtype}', a sequence of '{mtype}' or a sequence of tuples "
                         f"'(probability: float, '{mtype}')'.")

    parsed_probabilities = []
    parsed_moves = []

    for move in moves:
        if isinstance(move, Move) and moves.__class__.__name__ == mtype:
            parsed_probabilities.append(1.0)
            parsed_moves.append(move)

        elif isinstance(move, tuple):
            if len(move) == 2 and isinstance(move[0], float) and isinstance(move[1], Move) \
                    and moves.__class__.__name__ == mtype:
                parsed_probabilities.append(move[0])
                parsed_moves.append(move[1])

            else:
                raise ValueError(f"Invalid format for tuple '{move}', expected '(probability: float, Move)'.")

        else:
            raise ValueError(f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                             f"defining a move, expected a '{mtype}' or tuple '(probability: float, '{mtype}')'.")

    if sum(parsed_probabilities) != 1:
        _sum = sum(parsed_probabilities)
        parsed_probabilities = [proba / _sum for proba in parsed_probabilities]

    return parsed_probabilities, parsed_moves


def get_mean_cost(fun: Callable[[np.ndarray, Any], float],
                  x: np.ndarray,
                  _n: int, *args) -> float:
    """
    Get the mean of <n> function evaluations for vector of values <x>.

    :param fun: a function to evaluate.
    :param x: a vector of values.
    :param _n: the number of evaluations to compute.

    :return: the mean of function evaluations at x.
    """
    return float(np.mean([fun(x, *args) for _ in range(_n)]))


def acceptance_probability(current_cost: float,
                           new_cost: float,
                           _T: float) -> float:
    """
    Compute the acceptance probability for a new proposed cost, given the current cost and a temperature.

    :param current_cost: the current cost.
    :param new_cost: the new proposed cost.
    :param _T: the current temperature.

    :return: the probability of acceptance of the new proposed cost.
    """
    return (current_cost - new_cost) / _T


def sigma(k: int,
          T_0: float,
          alpha: float,
          epsilon: float) -> float:
    """
    Compute the estimated standard deviation at iteration k.

    :param k: the iteration number.
    :param T_0: initial temperature value.
    :param alpha: rate of temperature decrease.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)

    :return: the estimated standard deviation.
    """
    return T_0 * (alpha * (1 - epsilon)) ** k


def n(k: int,
      max_measures: int,
      sigma_max: float,
      T_0: float,
      alpha: float,
      epsilon: float) -> int:
    """
    Compute the number of necessary measures at iteration k.

    :param k: the iteration number.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param sigma_max: the maximal values (reached at iteration 0) for sigma.
    :param T_0: initial temperature value.
    :param alpha: rate of temperature decrease.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)

    :return: the number of necessary measures.
    """
    return int(np.ceil((max_measures * sigma_max ** 2) /
                       ((max_measures - 1) * sigma(k, T_0, alpha, epsilon) ** 2 + sigma_max ** 2)))


def T(k: int,
      T_0: float,
      alpha: float) -> float:
    """
    Compute the temperature at iteration k.

    :param k: the iteration number.
    :param T_0: initial temperature value.
    :param alpha: rate of temperature decrease.

    :return: the temperature.
    """
    return T_0 * alpha ** k
