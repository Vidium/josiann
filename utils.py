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
from multiprocessing import cpu_count
from dataclasses import dataclass, field
from plotly.subplots import make_subplots

from typing import Union, Sequence, Tuple, List, Callable, Any, Optional

from .name_utils import ShapeError


# ====================================================
# code
class Trace:
    """
    Object for storing the trace history of an SA run.

    :param nb_iterations: number of expected iterations for the SA algorithm.
    :param shape: shape of the matrices to store (nb_walkers, d).
    :param window_size: size of the window of values to test for convergence.
    """

    def __init__(self,
                 nb_iterations: int,
                 shape: Tuple[int, int],
                 window_size: int):
        self.__position_trace = np.zeros((nb_iterations, shape[0], shape[1]), dtype=np.float32)
        self.__cost_trace = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self.__temperature_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.__n_trace = np.zeros(nb_iterations, dtype=np.int32)
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
        self.__position_trace = self.__position_trace[:self.__iteration_counter+1]
        self.__cost_trace = self.__cost_trace[:self.__iteration_counter+1]
        self.__temperature_trace = self.__temperature_trace[:self.__iteration_counter+1]
        self.__n_trace = self.__n_trace[:self.__iteration_counter+1]
        self.__sigma_trace = self.__sigma_trace[:self.__iteration_counter+1]

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

    def are_stuck(self) -> List[bool]:
        """
        Detect which walkers are stuck at the same position within the last <window_size> positions.

        :return: The list of stuck walkers.
        """
        if self.__iteration_counter < self.__window_size:
            return [False for _ in range(self.nb_walkers)]

        return [np.sum(self.__accepted[self.__iteration_counter - self.__window_size:self.__iteration_counter, w]) == 0
                for w in range(self.nb_walkers)]

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
        if self.__iteration_counter < self.__window_size:
            return np.nan, np.nan, []

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
                                     name=f'Walker #{w}',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=[f"<b>Walker</b>: {w}<br>"
                                                f"<b>Cost</b>: {cost:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, cost in enumerate(self.__cost_trace[:, w])],
                                     hoverinfo="text",
                                     showlegend=True,
                                     legendgroup=f'Walker #{w}'), row=1, col=1)

        for i in range(self.ndim):
            for w in range(self.nb_walkers):
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self.__position_trace[:, w, i],
                                         marker=dict(color='rgba(0, 0, 0, 0.3)'),
                                         name=f'Walker #{w}',
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Position</b>: {self.__position_trace[iteration, w, i]:.4f}<br>"
                                                    f"<b>Cost</b>: {cost:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost in enumerate(self.__cost_trace[:, w])],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=i + 2, col=1)

            if true_values is not None:
                fig.add_trace(go.Scatter(x=[0, self.nb_positions-1],
                                         y=[true_values[i], true_values[i]],
                                         mode='lines',
                                         marker=dict(color='rgba(200, 0, 0, 1)'),
                                         name=f'True value',
                                         showlegend=False), row=i + 2, col=1)

                fig.add_annotation(
                    x=len(self.__position_trace),
                    y=np.max(self.__position_trace[:, :, i]),
                    xref=f"x{i + 2}",
                    yref=f"y{i + 2}",
                    text=f"True value : {true_values[i]}",
                    showarrow=False,
                    borderwidth=0,
                    borderpad=4,
                    bgcolor="#eb9a9a",
                    opacity=0.8
                )

        for i in range(self.ndim + 1):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

        fig['layout'].update(height=200 * (self.ndim + 1), width=600, margin=dict(t=40, b=10, l=10, r=10))

        fig.show()

    def plot_parameters(self) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.
        """
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Temperature", "sigma", "n", "Acceptance fraction (%)"),
                            vertical_spacing=0.05)

        fig.add_trace(go.Scatter(x=list(range(1, self.nb_iterations)),
                                 y=self.__temperature_trace,
                                 name='T',
                                 hovertext=[f"<b>Temperature</b>: {_T:.4f}<br>"
                                            f"<b>Iteration</b>: {iteration+1}"
                                            for iteration, _T in enumerate(self.__temperature_trace)],
                                 hoverinfo="text",
                                 showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(1, self.nb_iterations)),
                                 y=self.__sigma_trace,
                                 name='sigma',
                                 hovertext=[f"<b>Sigma</b>: {_sigma:.4f}<br>"
                                            f"<b>Iteration</b>: {iteration+1}"
                                            for iteration, _sigma in enumerate(self.__sigma_trace)],
                                 hoverinfo="text",
                                 showlegend=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=list(range(1, self.nb_iterations)),
                                 y=self.__n_trace,
                                 name='n',
                                 hovertext=[f"<b>Number evaluations</b>: {_n}<br>"
                                            f"<b>Iteration</b>: {iteration+1}"
                                            for iteration, _n in enumerate(self.__n_trace)],
                                 hoverinfo="text",
                                 showlegend=False), row=3, col=1)

        for w in range(self.nb_walkers):
            accepted_proportion = np.concatenate((np.array([np.nan for _ in range(self.__window_size)]),
                                                  np.convolve(self.__accepted[:, w],
                                                              np.ones(self.__window_size) / self.__window_size,
                                                              mode='valid') * 100))

            fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                                     y=accepted_proportion,
                                     name=f'Walker #{w}',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=[f"<b>Acceptance percentage</b>: {accepted}<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, accepted in enumerate(accepted_proportion)],
                                     hoverinfo="text",
                                     showlegend=True), row=4, col=1)

        fig.update_layout(yaxis4=dict(range=[0, 100]), height=150 * (self.ndim + 1), width=600,
                          margin=dict(t=40, b=10, l=10, r=10))

        for i in range(4):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

        fig.show()


@dataclass
class State:
    """
    Object for describing the current state of the SA algorithm.

    complementary_set: set of complementary vectors x_[k] of shape (nb_walkers-1, ndim)
    iteration: current iteration number.
    max_iter: maximum iteration number.
    """
    complementary_set: np.ndarray
    iteration: int
    max_iter: int


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
                     bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]],
                     nb_cores: int,
                     vectorized: bool) -> Tuple[
    Tuple, np.ndarray, int, int, float, float, float, float, bool, int
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
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
        or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
    :param nb_cores: number of cores that can be used to move walkers in parallel.
    :param vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
        just one. (<nb_cores> parameter will be set to 1 in this case.)

    :return: Valid parameters.
    """
    args = args if args is not None else ()

    if x0.ndim == 1:
        x0 = np.array([x0 + np.random.uniform(-0.5e-10, 0.5e-10) for _ in range(nb_walkers)])

    if x0.shape[0] != nb_walkers:
        raise ShapeError(f'Matrix of initial values should have {nb_walkers} rows (equal to the number of '
                         f'parallel walkers), not {x0.shape[0]}')

    if np.all([x0[0] == x0[i] for i in range(1, len(x0))]):
        warnings.warn('Initial positions are the same for all walkers, adding random noise.')

        x0 = np.array([x0[i] + np.random.uniform(-0.5e-10, 0.5e-10) for i in range(len(x0))])

    if bounds is not None:
        if isinstance(bounds, tuple) and isinstance(bounds[0], numbers.Number) \
                and isinstance(bounds[1], numbers.Number):
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
                        print(x0[:, dim_index])
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

    if vectorized:
        nb_cores = 1

    if nb_cores < 1:
        raise ValueError('Cannot use less than one core.')
    elif nb_cores > cpu_count():
        raise ValueError(f"Cannot use more than available CPUs ({cpu_count()}).")
    else:
        nb_cores = int(nb_cores)

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, computed_T_0, nb_cores


def get_mean_cost(fun: Callable[[np.ndarray, Any], float],
                  x: np.ndarray,
                  _n: int,
                  *args) -> float:
    """
    Get the mean of <n> function evaluations for vector of values <x>.

    :param fun: a function to evaluate.
    :param x: a vector of values.
    :param _n: the number of evaluations to compute.

    :return: the mean of function evaluations at x.
    """
    return float(np.mean([fun(x, *args)**2 for _ in range(_n)]))


def get_vectorized_mean_cost(fun: Callable[[np.ndarray, Any], List[float]],
                             x: np.ndarray,
                             _n: int,
                             *args) -> List[float]:
    """
    Same as 'get_mean_cost' but <fun> is a vectorized function and costs are computed for all walkers at once.

    :param fun: a vectorized function to evaluate.
    :param x: a matrix of position vectors of shape (nb_walkers, d).
    :param _n: the number of evaluations to compute.

    :return: the mean of function evaluations at x.
    """
    return list(np.mean([np.array(fun(x, *args))**2 for _ in range(_n)], axis=0))


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
