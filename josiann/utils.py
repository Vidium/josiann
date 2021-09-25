# coding: utf-8
# Created on 26/07/2021 12:08
# Author : matteo

# ====================================================
# imports
import numbers
import warnings
import collections.abc
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from multiprocessing import cpu_count
from dataclasses import dataclass, field
from plotly.subplots import make_subplots

from typing import Union, Sequence, Tuple, List, Callable, Optional

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
                 shape: Tuple[int, ...],
                 window_size: int):
        self.__position_trace = np.zeros((nb_iterations, shape[0], shape[1]), dtype=np.float32)
        self.__best_position_trace = np.zeros((nb_iterations, shape[1]+3), dtype=np.float32)
        self.__cost_trace = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self.__cost_trace_n = np.zeros((nb_iterations, shape[0]), dtype=np.int32)
        self.__temperature_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.__n_trace = np.zeros(nb_iterations, dtype=np.int32)
        self.__sigma_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.__accepted = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self.__rescued = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self.__computation_time = np.zeros(nb_iterations, dtype=np.float32)

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
        self.__cost_trace_n = np.concatenate((np.array([[0 for _ in range(len(position))]]),
                                              self.__cost_trace_n))

        best_index = np.argmin(costs)
        self.__best_position_trace = np.concatenate(([np.append(position.copy()[best_index],
                                                                [costs[best_index], best_index, 0])],
                                                     self.__best_position_trace))

    def finalize(self) -> None:
        """
        Cleanup traces at the end of the SA algorithm.
        """
        self.__position_trace = self.__position_trace[:self.position_counter]
        self.__cost_trace = self.__cost_trace[:self.position_counter]
        self.__cost_trace_n = self.__cost_trace_n[:self.position_counter]
        self.__temperature_trace = self.__temperature_trace[:self.__iteration_counter]
        self.__n_trace = self.__n_trace[:self.__iteration_counter]
        self.__sigma_trace = self.__sigma_trace[:self.__iteration_counter]
        self.__accepted = self.__accepted[:self.__iteration_counter]
        self.__rescued = self.__rescued[:self.__iteration_counter]
        self.__computation_time = self.__computation_time[:self.__iteration_counter]

    def store(self,
              position: np.ndarray,
              costs: List[float],
              temperature: float,
              _n: int,
              _sigma: float,
              accepted: List[bool]) -> int:
        """
        Save the current position of the vector to optimize, the current cost, temperature and number of averaged
            function evaluations.

        :param position: the current vector.
        :param costs: the current costs.
        :param temperature: the current temperature.
        :param _n: the current number of averaged function evaluations.
        :param _sigma: the current estimated standard deviation.
        :param accepted: were the current propositions accepted ?

        :return: the index at which was stored the data.
        """
        self.__position_trace[self.position_counter] = position.copy()
        self.__cost_trace[self.position_counter] = np.array(costs)
        self.__temperature_trace[self.__iteration_counter] = float(temperature)
        self.__n_trace[self.__iteration_counter] = int(_n)
        self.__sigma_trace[self.__iteration_counter] = float(_sigma)
        self.__accepted[self.__iteration_counter] = np.array(accepted)

        accepted_n = np.ones(self.nb_walkers) * _n
        rejected_mask = np.where(~np.array(accepted))[0]
        accepted_n[rejected_mask] = self.__cost_trace_n[self.position_counter - 1, rejected_mask]
        self.__cost_trace_n[self.position_counter] = accepted_n

        self.__iteration_counter += 1
        return self.__iteration_counter - 1

    def update(self,
               index: int,
               position: np.ndarray,
               costs: List[float],
               rescued: List[bool],
               best_position: np.ndarray,
               best_cost: float,
               best_index: List[int],
               computation_time: float) -> None:
        """


        :param index: index at which to update the data.
        :param position: the current vector.
        :param costs: the current costs.
        :param rescued: were the walkers rescued at this iteration ?
        :param best_position: best position vector reached since the start of the SA algorithm.
        :param best_cost: cost associated to the best position.
        :param best_index: tuple (iteration, walker index) associated to the best position.
        :param computation_time: time it took to compute this iteration.
        """
        position_index = index + 1 * self.__initialized

        if np.any(rescued):
            rescued_mask = np.where(rescued)[0]

            self.__position_trace[position_index, rescued_mask] = position.copy()[rescued_mask]
            self.__cost_trace[position_index, rescued_mask] = np.array(costs)[rescued_mask]

            self.__rescued[index] = np.array(rescued)

            rescued_n = self.__cost_trace_n[position_index]
            rescued_n[rescued_mask] = self.__cost_trace_n[best_index[0], best_index[1]]
            self.__cost_trace_n[position_index] = rescued_n

        self.__best_position_trace[position_index, :self.ndim] = best_position
        self.__best_position_trace[position_index, self.ndim] = best_cost
        self.__best_position_trace[position_index, self.ndim + 1] = best_index[1]
        self.__best_position_trace[position_index, self.ndim + 2] = self.__n_trace[best_index[0]-1]

        self.__computation_time[index] = computation_time

    def reached_convergence(self,
                            tolerance: float) -> bool:
        """
        Has the cost trace reached convergence within a tolerance margin ?

        :param tolerance: the allowed root mean square deviation.

        :return: Whether the cost trace has converged.
        """
        if self.__iteration_counter < self.__window_size:
            return False

        # in case of rescue, if all workers end up on the same position, consider it as convergence
        if np.any(self.__rescued[self.__iteration_counter - 1]):
            if np.all(np.all(self.__position_trace[self.position_counter - 1] ==
                             self.__position_trace[self.position_counter - 1][0], axis=1)):
                return True

        mean_window = np.mean(self.__cost_trace[self.position_counter - self.__window_size:self.position_counter])
        RMSD = np.sqrt(np.sum(
            (self.__cost_trace[self.position_counter - self.__window_size:self.position_counter] - mean_window) ** 2
        ) / (self.__window_size - 1))

        return RMSD < tolerance

    def mean_acceptance_fraction(self) -> float:
        """
        Get the mean proportion of accepted proposition in the last <window_size> propositions over all walkers.

        :return: The mean proportion of accepted proposition in the last <window_size> propositions over all walkers.
        """
        if self.__iteration_counter < self.__window_size:
            return np.nan

        return float(np.mean(
            [np.sum(self.__accepted[self.__iteration_counter - self.__window_size:self.__iteration_counter, w]) /
             self.__window_size for w in range(self.nb_walkers)]))

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
        return self.__temperature_trace.shape[0]

    @property
    def position_counter(self) -> int:
        return self.__iteration_counter + 1 * self.__initialized

    def get_best(self) -> Tuple[np.ndarray, float, List[int]]:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        :return: the best vector, best cost and iteration number that reached it.
        """
        START = 0 + 1 * self.__initialized if self.position_counter > 1 else 0

        lookup_array = self.__cost_trace[
                       max(START, self.position_counter - self.__window_size):self.position_counter].copy()

        # normalize lookup array to account for variance drop
        current_n = self.__n_trace[self.__iteration_counter-1]
        with np.errstate(divide='ignore'):
            correction_factors = current_n / self.__cost_trace_n[max(START, self.position_counter -
                                                                     self.__window_size):self.position_counter]

        lookup_array *= correction_factors

        _best_index = list(np.unravel_index(np.argmin(lookup_array), lookup_array.shape))

        _best_index[0] += max(START, self.position_counter - self.__window_size)
        corrected_index = _best_index[0]

        while self.__cost_trace[corrected_index, _best_index[1]] == self.__cost_trace[_best_index[0], _best_index[1]]:
            corrected_index -= 1
            if corrected_index < 0:
                break

        _best_index[0] = corrected_index + 1 * self.__initialized

        return self.__position_trace[_best_index[0], _best_index[1]], \
            self.__cost_trace[_best_index[0], _best_index[1]], \
            _best_index

    def get_position_trace(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get traces for vector, cost values, n at cost evaluation and best vector along iterations.

        :return: Traces for vector, cost values, n at cost evaluation and best vector along iterations.
        """
        return self.__position_trace, self.__cost_trace, self.__cost_trace_n, self.__best_position_trace

    def get_parameters_trace(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get traces related to parameters values along iterations.

        :return: Traces related to parameters values along iterations.
        """
        return self.__temperature_trace, self.__n_trace, self.__sigma_trace

    def get_acceptance_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get traces related to acceptance rates along iterations.

        :return: Traces related to acceptance rates along iterations.
        """
        return self.__accepted, self.__rescued

    def plot_positions(self,
                       true_values: Optional[Sequence[float]] = None,
                       extended: bool = False,
                       show: bool = True,
                       save: Optional[Path] = None) -> None:
        """
        Plot reached positions and costs for the vector to optimize along iterations.

        :param true_values: an optional sequence of known true values for each dimension of the vector to optimize.
        :param extended: plot additional plots ? (mostly for debugging)
        :param show: render the plot ? (default True)
        :param save: optional path to save the plot as an html file.
        """
        if true_values is not None and len(true_values) != self.ndim:
            raise ShapeError(f'The vector of true values should have {self.ndim} dimensions, not {len(true_values)}.')

        supp_plots = 3 if extended else 1
        titles = ["Costs"] + [f'Dimension {i}' for i in range(self.ndim)]
        if extended:
            titles.insert(1, "n at cost evaluation")
            titles.insert(2, "Best cost evolution")

        fig = make_subplots(rows=self.ndim + supp_plots, cols=1, shared_xaxes=True,
                            subplot_titles=titles,
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

            if extended:
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self.__cost_trace_n[:, w],
                                         name=f'Walker #{w}',
                                         marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>n at cost evaluation</b>: {cost_n}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost_n in enumerate(self.__cost_trace_n[:, w])],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=2, col=1)

        if extended:
            with np.errstate(divide='ignore', invalid='ignore'):
                best_costs = [self.__best_position_trace[it, -3] * np.insert(self.__n_trace, 0, 0)[it] /
                              self.__best_position_trace[it, -1]
                              for it in range(self.nb_positions)]
            best_walkers = [int(self.__best_position_trace[it, -2]) for it in range(self.nb_positions)]

            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=best_costs,
                                     name='Best cost evolution',
                                     marker=dict(color='rgba(252, 196, 25, 1.)'),
                                     hovertext=[f"<b>Walker</b>: {best_walkers[iteration]}<br>"
                                                f"<b>Cost</b>: {cost}<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, cost in enumerate(best_costs)],
                                     hoverinfo="text",
                                     showlegend=False), row=3, col=1)

        for d in range(self.ndim):
            for w in range(self.nb_walkers):
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self.__position_trace[:, w, d],
                                         marker=dict(color='rgba(0, 0, 0, 0.3)'),
                                         name=f'Walker #{w}',
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Position</b>: {self.__position_trace[iteration, w, d]:.4f}<br>"
                                                    f"<b>Cost</b>: {cost:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost in enumerate(self.__cost_trace[:, w])],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=d + supp_plots + 1, col=1)

                # add rescue points
                rescue_iterations = np.where(self.__rescued[:, w])[0]
                fig.add_trace(go.Scatter(x=rescue_iterations,
                                         y=self.__position_trace[rescue_iterations, w, d],
                                         mode='markers',
                                         marker=dict(color='rgba(0, 255, 0, 0.3)',
                                                     symbol=2,
                                                     size=10),
                                         name=f'Rescues for walker #{w}',
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Position</b>: {self.__position_trace[iteration, w, d]:.4f}<br>"
                                                    f"<b>Cost</b>: {self.__cost_trace[iteration, w]:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration in rescue_iterations],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=d + supp_plots + 1, col=1)

            # add best points
            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=self.__best_position_trace[:, d],
                                     mode='markers',
                                     marker=dict(color='rgba(252, 196, 25, 1.)',
                                                 symbol=0,
                                                 size=3),
                                     name='Best cost',
                                     hovertext=[f"<b>Walker</b>: {int(self.__best_position_trace[iteration, -2])}<br>"
                                                f"<b>Position</b>: {position:.4f}<br>"
                                                f"<b>Cost</b>: {self.__best_position_trace[iteration, -3]:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, position in enumerate(self.__best_position_trace[:, d])],
                                     hoverinfo="text",
                                     showlegend=True if d == 0 else False,
                                     legendgroup='Best cost'), row=d + supp_plots + 1, col=1)

            if true_values is not None:
                fig.add_trace(go.Scatter(x=[0, self.nb_positions-1],
                                         y=[true_values[d], true_values[d]],
                                         mode='lines',
                                         marker=dict(color='rgba(200, 0, 0, 1)'),
                                         line=dict(dash='dash'),
                                         name='True value',
                                         showlegend=False), row=d + supp_plots + 1, col=1)

                fig.add_annotation(
                    x=self.nb_positions-1,
                    y=np.max(self.__position_trace[:, :, d]),
                    xref=f"x{d + supp_plots + 1}",
                    yref=f"y{d + supp_plots + 1}",
                    text=f"True value : {true_values[d]}",
                    showarrow=False,
                    borderwidth=0,
                    borderpad=4,
                    bgcolor="#eb9a9a",
                    opacity=0.8
                )

        for i in range(self.ndim + 1):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

        fig['layout'].update(height=200 * (self.ndim + 2), width=600, margin=dict(t=40, b=10, l=10, r=10),
                             xaxis_range=[0, self.nb_positions-1])

        if show:
            fig.show()

        if save is not None:
            fig.write_html(str(save))

    def plot_parameters(self,
                        extended: bool = False,
                        show: bool = True,
                        save: Optional[Path] = None) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.

        :param extended: plot additional plots ? (mostly for debugging)
        :param show: render the plot ? (default True)
        :param save: optional path to save the plot as an html file.
        """
        sub_plots = 5 if extended else 4
        titles = ["Temperature", "sigma", "n", "Acceptance fraction (%)"]
        if extended:
            titles.append("Computation time (s)")

        fig = make_subplots(rows=sub_plots, cols=1, shared_xaxes=True,
                            subplot_titles=titles,
                            vertical_spacing=0.05)

        if self.nb_iterations:
            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=self.__temperature_trace,
                                     name='T',
                                     hovertext=[f"<b>Temperature</b>: {_T:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration+1}"
                                                for iteration, _T in enumerate(self.__temperature_trace)],
                                     hoverinfo="text",
                                     showlegend=False), row=1, col=1)

            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=self.__sigma_trace,
                                     name='sigma',
                                     hovertext=[f"<b>Sigma</b>: {_sigma:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration+1}"
                                                for iteration, _sigma in enumerate(self.__sigma_trace)],
                                     hoverinfo="text",
                                     showlegend=False), row=2, col=1)

            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=self.__n_trace,
                                     name='n',
                                     hovertext=[f"<b>Number evaluations</b>: {_n}<br>"
                                                f"<b>Iteration</b>: {iteration+1}"
                                                for iteration, _n in enumerate(self.__n_trace)],
                                     hoverinfo="text",
                                     showlegend=False), row=3, col=1)

            accepted_proportions = np.zeros((self.nb_walkers, self.position_counter))

            for w in range(self.nb_walkers):
                accepted_proportions[w] = np.concatenate((np.array([np.nan for _ in range(self.__window_size)]),
                                                          np.convolve(self.__accepted[:, w],
                                                                      np.ones(self.__window_size) / self.__window_size,
                                                                      mode='valid') * 100))

                fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                         y=accepted_proportions[w],
                                         name=f'Walker #{w}',
                                         marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Acceptance percentage</b>: {accepted:.2f}%<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, accepted in enumerate(accepted_proportions[w])],
                                         hoverinfo="text",
                                         showlegend=True), row=4, col=1)

            mean_acceptance_proportions = np.mean(accepted_proportions, axis=0)
            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=mean_acceptance_proportions,
                                     name='Mean acceptance',
                                     marker=dict(color='rgba(33, 33, 99, 1.)'),
                                     hovertext=[f"<b>Mean acceptance percentage</b>: {accepted:.2f}%<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, accepted in enumerate(mean_acceptance_proportions)],
                                     hoverinfo="text",
                                     showlegend=True), row=4, col=1)

            if extended:
                fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                         y=self.__computation_time,
                                         name='T',
                                         hovertext=[f"<b>Time</b>: {_time:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration + 1}"
                                                    for iteration, _time in enumerate(self.__computation_time)],
                                         hoverinfo="text",
                                         showlegend=False), row=5, col=1)

        fig.update_layout(yaxis4=dict(range=[0, 100]), height=150 * (self.ndim + 1), width=600,
                          margin=dict(t=40, b=10, l=10, r=10))

        for i in range(4):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

        if show:
            fig.show()

        if save is not None:
            fig.write_html(str(save))


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
    sigma_max: float
    tol: float
    window_size: int
    alpha: float
    T_final: float
    nb_cores: int
    vectorized: bool
    active_backup: bool
    x: np.ndarray = field(init=False)
    x_cost: float = field(init=False)
    x_iter: List[int] = field(init=False)

    def __post_init__(self):
        try:
            self.x, self.x_cost, self.x_iter = self.trace.get_best()
        except Exception:
            self.x, self.x_cost, self.x_iter = np.nan, np.nan, [np.nan, np.nan]

    def __repr__(self) -> str:
        return f"Message : {self.message}\n" \
               f"User parameters : \n" \
               f"\targs : {self.args}\n" \
               f"\tx0 : {self.x0}\n" \
               f"\tmax_iter : {self.max_iter}\n" \
               f"\tmax_measures : {self.max_measures}\n" \
               f"\tfinal_acceptance_probability : {self.final_acceptance_probability}\n" \
               f"\tepsilon : {self.epsilon}\n" \
               f"\tT_0 : {self.T_0}\n" \
               f"\tsigma_max : {self.sigma_max}\n" \
               f"\ttol : {self.tol}\n" \
               f"\tNb cores : {self.active_backup}\n"\
               f"\tVectorized : {self.vectorized}\n" \
               f"\tUsed storage : {self.active_backup}\n" \
               f"Computed parameters : \n" \
               f"\talpha : {self.alpha}\n" \
               f"\tT_final : {self.T_final}\n" \
               f"\twindow_size : {self.window_size}\n" \
               f"Success : {self.success}\n" \
               f"Lowest cost : {self.x_cost} (reached at iteration {self.x_iter[0]+1} by walker #{self.x_iter[1]})\n" \
               f"x: {self.x}"


def check_parameters(args: Optional[Sequence],
                     x0: np.ndarray,
                     nb_walkers: int,
                     max_iter: int,
                     max_measures: int,
                     final_acceptance_probability: float,
                     epsilon: float,
                     T_0: float,
                     tol: float,
                     bounds: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]],
                     nb_cores: int,
                     vectorized: bool) -> Tuple[
    Tuple, np.ndarray, int, int, float, float, float, float, int
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
    :param T_0: initial temperature value.
    :param tol: the convergence tolerance.
    :param bounds: an optional sequence of bounds (one for each <n> dimensions) with the following format:
        (lower_bound, upper_bound)
        or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.
    :param nb_cores: number of cores that can be used to move walkers in parallel.
    :param vectorized: if True, the cost function <fun> is expected to work on an array of position vectors instead of
        just one. (<nb_cores> parameter will be set to 1 in this case.)

    :return: Valid parameters.
    """
    args = tuple(args) if args is not None else ()

    if x0.ndim == 1:
        if len(x0) > 1:
            x0 = np.array([x0 + np.random.uniform(-0.5e-10, 0.5e-10) for _ in range(nb_walkers)])

        else:
            x0 = np.array([x0])

    if x0.shape[0] != nb_walkers:
        raise ShapeError(f'Matrix of initial values should have {nb_walkers} rows (equal to the number of '
                         f'parallel walkers), not {x0.shape[0]}')

    if len(x0) > 1 and np.all([x0[0] == x0[i] for i in range(1, len(x0))]):
        warnings.warn('Initial positions are the same for all walkers, adding random noise.')

        x0 = np.array([x0[i] + np.random.uniform(-0.5e-10, 0.5e-10) for i in range(len(x0))])

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
            raise TypeError("'bounds' parameter must be an optional sequence of bounds (one for each <n> dimensions) "
                            "with the following format: \n"
                            "\t(lower_bound, upper_bound)\n "
                            "or a single (lower_bound, upper_bound) tuple of bounds to set for all dimensions.")

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

    if T_0 < 0:
        raise ValueError("'T_0' parameter must be at least 0.")
    else:
        T_0 = float(T_0)

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

    return args, x0, max_iter, max_measures, final_acceptance_probability, epsilon, T_0, tol, nb_cores


def get_mean_cost(fun: Callable,
                  x: np.ndarray,
                  _n: int,
                  args: Tuple,
                  previous_evaluations: Tuple[int, float]) -> float:
    """
    Get the mean of <n> function evaluations for vector of values <x>.

    :param fun: a function to evaluate.
    :param x: a vector of values.
    :param _n: the number of evaluations to compute.
    :param args: arguments to be passed to <fun>.
    :param previous_evaluations: previously computed function evaluations at position x: number of last function
        evaluations and obtained mean.

    :return: the mean of function evaluations at x.
    """
    last_n, last_mean = previous_evaluations
    return last_mean * last_n / _n + sum([fun(x, *args)**2 for _ in range(_n - last_n)]) / _n


def get_vectorized_mean_cost(fun: Callable,
                             x: np.ndarray,
                             _n: int,
                             args: Tuple,
                             previous_evaluations: List[Tuple[int, float]]) -> List[float]:
    """
    Same as 'get_mean_cost' but <fun> is a vectorized function and costs are computed for all walkers at once.

    :param fun: a vectorized function to evaluate.
    :param x: a matrix of position vectors of shape (nb_walkers, d).
    :param _n: the number of evaluations to compute.
    :param args: arguments to be passed to <fun>.
    :param previous_evaluations: list of previously computed function evaluations at position x: number of last function
        evaluations and obtained means for each walker position.

    :return: the mean of function evaluations at x.
    """
    # TODO : compare speed when vectorizing on walkers instead of n
    evaluations = [0. for _ in range(len(x))]

    for walker_index in range(len(x)):
        last_n, last_mean = previous_evaluations[walker_index]
        remaining_n = _n - last_n
        evaluations[walker_index] = last_mean * last_n / _n + \
            sum(fun(np.tile(x[walker_index], (remaining_n, 1)), *args)) / _n

    return evaluations


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
