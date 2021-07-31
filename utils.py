# coding: utf-8
# Created on 26/07/2021 12:08
# Author : matteo

# ====================================================
# imports
import numbers
import warnings
import collections
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field
from plotly.subplots import make_subplots

from typing import Union, Sequence, Tuple, List, Callable, Any, Optional

from .name_utils import ShapeError
from .moves import Move, EnsembleMove


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
        self.__position_trace = np.zeros((nb_iterations, shape[0], shape[1]), dtype=np.float32)
        self.__cost_trace = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self.__temperature_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.__n_trace = np.zeros(nb_iterations, dtype=np.int16)
        self.__sigma_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.__accepted = np.zeros((nb_iterations, shape[0]), dtype=np.float32)

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

        self.__position_trace = np.concatenate((np.array([position.copy()]), self.__position_trace))
        self.__cost_trace = np.concatenate((np.array([costs]), self.__cost_trace))

    def finalize(self) -> None:
        """
        Cleanup traces at the end of the SA algorithm.
        """
        self.__position_trace = self.__position_trace[:self.__iteration_counter]
        self.__cost_trace = self.__cost_trace[:self.__iteration_counter]
        self.__temperature_trace = self.__temperature_trace[:self.__iteration_counter]
        self.__n_trace = self.__n_trace[:self.__iteration_counter]
        self.__sigma_trace = self.__sigma_trace[:self.__iteration_counter]

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
        self.__position_trace[self.__iteration_counter + 1 * self.__initialized] = position.copy()
        self.__cost_trace[self.__iteration_counter + 1 * self.__initialized] = np.array(costs)
        self.__temperature_trace[self.__iteration_counter] = float(temperature)
        self.__n_trace[self.__iteration_counter] = int(_n)
        self.__sigma_trace[self.__iteration_counter] = float(_sigma)
        self.__accepted[self.__iteration_counter] = np.array(accepted)

        self.__iteration_counter += 1

    def reached_convergence(self,
                            tolerance: float) -> bool:
        """
        Has the cost trace reached convergence within a tolerance margin ?

        :param tolerance: the allowed root mean square deviation.

        :return: Whether the cost trace has converged.
        """
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
            return [np.nan for _ in range(self.nb_walkers)]

        return [np.sum(self.__accepted[self.__iteration_counter - self.__window_size:self.__iteration_counter, w]) /
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

    def get_best(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        :return: the best vector, best cost and iteration number that reached it.
        """
        lookup_array = self.__cost_trace[max(0, self.__iteration_counter - self.__window_size):self.__iteration_counter]

        _best_index = list(np.unravel_index(np.argmin(lookup_array), lookup_array.shape))

        _best_index[0] += max(0, self.__iteration_counter - self.__window_size)

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

        fig = make_subplots(rows=self.ndim + 1, cols=1, shared_xaxes=True,
                            subplot_titles=("Costs", *(f'Dimension {i}' for i in range(self.ndim))),
                            vertical_spacing=0.05)

        for w in range(self.nb_walkers):
            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=self.__cost_trace[:, w],
                                     name=f'costs[{w}]',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=self.__cost_trace,
                                     showlegend=False), row=1, col=1)

        for i in range(self.ndim):
            for w in range(self.nb_walkers):
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self.__position_trace[:, w, i],
                                         marker=dict(color='rgba(0, 0, 0, 0.3)'),
                                         name=f'Walker #{w}',
                                         showlegend=False), row=i + 2, col=1)

            if true_values is not None:
                fig.add_trace(go.Scatter(x=[0, self.nb_positions],
                                         y=[true_values[i], true_values[i]],
                                         mode='lines',
                                         marker=dict(color='rgba(200, 0, 0, 1)'),
                                         name=f'True value',
                                         showlegend=False), row=i + 2, col=1)

                fig.add_annotation(
                    x=len(self.__position_trace),
                    y=np.max(self.__position_trace[:, :, i]),
                    xref=f"x{i+2}",
                    yref=f"y{i+2}",
                    text=f"True value : {true_values[i]}",
                    showarrow=False,
                    borderwidth=0,
                    borderpad=4,
                    bgcolor="#eb9a9a",
                    opacity=0.8
                )

        for i in range(self.ndim + 1):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

        fig['layout'].update(height=200*(self.ndim + 1), width=600, margin=dict(t=40, b=10, l=10, r=10))

        fig.show()

    def plot_parameters(self) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.
        """
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Temperature", "Sigma", "n", "Acceptance fraction (%)"),
                            vertical_spacing=0.05)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__temperature_trace,
                                 name='T',
                                 hovertext=self.__temperature_trace,
                                 showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__sigma_trace,
                                 name='sigma',
                                 hovertext=self.__sigma_trace,
                                 showlegend=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                 y=self.__n_trace,
                                 name='n',
                                 hovertext=self.__n_trace,
                                 showlegend=False), row=3, col=1)

        for w in range(self.nb_walkers):
            accepted_proportion = np.concatenate((np.array([np.nan for _ in range(self.__window_size)]),
                                                  np.convolve(self.__accepted[:, w],
                                                              np.ones(self.__window_size) / self.__window_size,
                                                              mode='valid') * 100))

            fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                     y=accepted_proportion,
                                     name=f'% accepted[{w}]',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=accepted_proportion,
                                     showlegend=False), row=4, col=1)

        fig.update_layout(yaxis4=dict(range=[0, 100]), height=150 * (self.ndim + 1), width=600,
                          margin=dict(t=40, b=10, l=10, r=10))

        for i in range(4):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

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
    x_iter: List[int] = field(init=False)

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
               f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter[0]} by walker #{self.x_iter[1]})\n" \
               f"x: {self.x}"


def parse_moves(moves: Union[Move, Sequence[Move], Sequence[Tuple[float, Move]]],
                nb_walkers: int) -> Tuple[List[float], List[Move]]:
    """
    Parse moves given by the user to obtain a list of moves and associated probabilities of drawing those moves.

    :param moves: a single Move object, a sequence of Moves (uniform probabilities are assumed on all Moves) or a
        sequence of tuples with format (probability: float, Move).
    :param nb_walkers: the number of parallel walkers in the ensemble.

    :return: the list of probabilities and the list of associated moves.
    """
    if not isinstance(moves, collections.Sequence) or isinstance(moves, str):
        if isinstance(moves, Move):
            if issubclass(type(moves), EnsembleMove) and nb_walkers < 2:
                raise ValueError('Ensemble moves require at least 2 walkers to be used.')

            return [1.0], [moves]

        raise ValueError(f"Invalid object '{moves}' of type '{type(moves)}' for defining moves, expected a "
                         f"'Move', a sequence of 'Move's or a sequence of tuples "
                         f"'(probability: float, 'Move')'.")

    parsed_probabilities = []
    parsed_moves = []

    for move in moves:
        if isinstance(move, Move):
            if issubclass(type(move), EnsembleMove) and nb_walkers < 2:
                raise ValueError('Ensemble moves require at least 2 walkers to be used.')

            parsed_probabilities.append(1.0)
            parsed_moves.append(move)

        elif isinstance(move, tuple):
            if len(move) == 2 and isinstance(move[0], float) and isinstance(move[1], Move):
                if issubclass(type(move[1]), EnsembleMove) and nb_walkers < 2:
                    raise ValueError('Ensemble moves require at least 2 walkers to be used.')

                parsed_probabilities.append(move[0])
                parsed_moves.append(move[1])

            else:
                raise ValueError(f"Invalid format for tuple '{move}', expected '(probability: float, Move)'.")

        else:
            raise ValueError(f"Invalid object '{move}' of type '{type(move)}' encountered in the sequence of moves for "
                             f"defining a move, expected a 'Move' or tuple '(probability: float, 'Move')'.")

    if sum(parsed_probabilities) != 1:
        _sum = sum(parsed_probabilities)
        parsed_probabilities = [proba / _sum for proba in parsed_probabilities]

    return parsed_probabilities, parsed_moves


def get_delta_max() -> float:
    """TODO"""
    raise NotImplementedError


def check_parameters(args: Optional[Sequence],
                     x0: np.ndarray,
                     nb_walkers: int,
                     max_iter: int,
                     max_measures: int,
                     final_acceptance_probability: float,
                     epsilon: float,
                     T_0: Optional[float],
                     tol: float,
                     window_size: int,
                     bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]]) -> Tuple[
    Tuple, np.ndarray, int, int, float, float, float, float, int, bool
]:
    """
    Check validity of parameters.

    :param args: an optional sequence of arguments to pass to the function to minimize.
    :param x0: a <d> dimensional vector of initial values.
    :param nb_walkers: the number of parallel walkers in the ensemble.
    :param max_iter: the maximum number of iterations before stopping the algorithm.
    :param max_measures: the maximum number of function evaluations to average per step.
    :param final_acceptance_probability: the targeted final acceptance probability at iteration <max_iter>.
    :param epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease (bigger values yield
        steeper descent profiles)
    :param T_0: optional initial temperature value.
    :param tol: the convergence tolerance.
    :param window_size: a window of the last <window_size> cost values are used to test for convergence.
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
        or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.

    :return: Valid parameters.
    """
    args = args if args is not None else ()

    if x0.ndim == 1:
        x0 = np.array([x0 for _ in range(nb_walkers)])

    if x0.shape[0] != nb_walkers:
        raise ShapeError(f'Matrix of initial values should have {nb_walkers} rows (equal to the number of '
                         f'parallel walkers), not {x0.shape[0]}')

    if bounds is not None:
        if isinstance(bounds, tuple) and isinstance(bounds[0], numbers.Number) and isinstance(bounds[1], numbers.Number):
            if np.any(x0 < bounds[0]) or np.any(x0 > bounds[1]):
                raise ValueError('Some values in x0 do not lie in between defined bounds.')

        elif isinstance(bounds, collections.Sequence):
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
                        f"'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                        f"with the following format: \n"
                        f"\t(lower_bound, upper_bound)\n "
                        f"or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.")

        else:
            raise TypeError(f"'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                            f"with the following format: \n"
                            f"\t(lower_bound, upper_bound)\n "
                            f"or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.")

    if max_iter < 0:
        raise ValueError("'max_iter' parameter must be positive.")
    else:
        max_iter = int(max_iter)

    if max_measures < 0:
        raise ValueError("'max_measures' parameter must be positive.")
    else:
        max_measures = int(max_measures)

    if final_acceptance_probability < 0 or final_acceptance_probability > 1:
        raise ValueError(f"Invalid value '{final_acceptance_probability}' for 'final_acceptance_probability', "
                         f"should be in [0, 1].")

    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"Invalid value '{epsilon}' for 'epsilon', should be in (0, 1).")

    if T_0 is not None and T_0 < 0:
        raise ValueError("'T_0' parameter must be at least 0.")

    if T_0 is None:
        T_0 = -get_delta_max() / np.log(0.8)
        computed_T_0 = True
    else:
        T_0 = float(T_0)
        computed_T_0 = False

    if tol <= 0:
        raise ValueError("'tol' parameter must be strictly positive.")

    if window_size < 1:
        raise ValueError("'window_size' parameter must be greater than 0.")
    else:
        window_size = int(window_size)

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, window_size, computed_T_0


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


def acceptance_log_probability(current_cost: float,
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
