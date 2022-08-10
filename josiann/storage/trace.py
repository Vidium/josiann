# coding: utf-8
# Created on 03/02/2022 10:40
# Author : matteo

# ====================================================
# imports
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

from typing import Optional, Sequence

from josiann.name_utils import ShapeError


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
                 shape: tuple[int, ...],
                 window_size: int,
                 detect_convergence: bool):
        self._position_trace = np.zeros((nb_iterations, shape[0], shape[1]), dtype=np.float32)
        self._best_position_trace = np.zeros((nb_iterations, shape[1]+3), dtype=np.float32)
        self._cost_trace = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self._cost_trace_n = np.zeros((nb_iterations, shape[0]), dtype=np.int32)
        self._temperature_trace = np.zeros(nb_iterations, dtype=np.float32)
        self._n_trace = np.zeros(nb_iterations, dtype=np.int32)
        self._sigma_trace = np.zeros(nb_iterations, dtype=np.float32)
        self._accepted = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self._rescued = np.zeros((nb_iterations, shape[0]), dtype=np.float32)
        self._computation_time = np.zeros(nb_iterations, dtype=np.float32)

        self._initialized = False
        self._detect_convergence = detect_convergence
        self._window_size = window_size
        self._iteration_counter = 0

    def __repr__(self) -> str:
        return f"Trace of {self._iteration_counter} iteration(s), {self.nb_walkers} walker(s) and " \
               f"{self.ndim} dimension(s)."

    def initialize(self,
                   position: np.ndarray,
                   costs: list[float]) -> None:
        """
        Save state zero before running the SA algorithm. This function should be called before actually storing run
            values.

        :param position: the initial vector.
        :param costs: the initial costs.
        """
        self._initialized = True

        self._position_trace = np.concatenate((np.array([position.copy()]), self._position_trace))
        self._cost_trace = np.concatenate((np.array([costs]), self._cost_trace))
        self._cost_trace_n = np.concatenate((np.array([[0 for _ in range(len(position))]]),
                                              self._cost_trace_n))

        best_index = np.argmin(costs)
        self._best_position_trace = np.concatenate(([np.append(position.copy()[best_index],
                                                                [costs[best_index], best_index, 0])],
                                                     self._best_position_trace))

    def finalize(self) -> None:
        """
        Cleanup traces at the end of the SA algorithm.
        """
        self._position_trace = self._position_trace[:self.position_counter]
        self._best_position_trace = self._best_position_trace[:self.position_counter]
        self._cost_trace = self._cost_trace[:self.position_counter]
        self._cost_trace_n = self._cost_trace_n[:self.position_counter]
        self._temperature_trace = self._temperature_trace[:self._iteration_counter]
        self._n_trace = self._n_trace[:self._iteration_counter]
        self._sigma_trace = self._sigma_trace[:self._iteration_counter]
        self._accepted = self._accepted[:self._iteration_counter]
        self._rescued = self._rescued[:self._iteration_counter]
        self._computation_time = self._computation_time[:self._iteration_counter]

    def store(self,
              position: np.ndarray,
              costs: list[float],
              temperature: float,
              _n: int,
              _sigma: float,
              accepted: list[bool]) -> int:
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
        self._position_trace[self.position_counter] = position.copy()
        self._cost_trace[self.position_counter] = np.array(costs)
        self._temperature_trace[self._iteration_counter] = float(temperature)
        self._n_trace[self._iteration_counter] = int(_n)
        self._sigma_trace[self._iteration_counter] = float(_sigma)
        self._accepted[self._iteration_counter] = np.array(accepted)

        accepted_n = np.ones(self.nb_walkers) * _n
        rejected_mask = np.where(~np.array(accepted))[0]
        accepted_n[rejected_mask] = self._cost_trace_n[self.position_counter - 1, rejected_mask]
        self._cost_trace_n[self.position_counter] = accepted_n

        self._iteration_counter += 1
        return self._iteration_counter - 1

    def update(self,
               index: int,
               position: np.ndarray,
               costs: list[float],
               rescued: list[bool],
               best_position: np.ndarray,
               best_cost: float,
               best_index: list[int],
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
        position_index = index + 1 * self._initialized

        if np.any(rescued):
            rescued_mask = np.where(rescued)[0]

            self._position_trace[position_index, rescued_mask] = position.copy()[rescued_mask]
            self._cost_trace[position_index, rescued_mask] = np.array(costs)[rescued_mask]

            self._rescued[index] = np.array(rescued)

            rescued_n = self._cost_trace_n[position_index]
            rescued_n[rescued_mask] = self._cost_trace_n[best_index[0], best_index[1]]
            self._cost_trace_n[position_index] = rescued_n

        self._best_position_trace[position_index, :self.ndim] = best_position
        self._best_position_trace[position_index, self.ndim] = best_cost
        self._best_position_trace[position_index, self.ndim + 1] = best_index[1]
        self._best_position_trace[position_index, self.ndim + 2] = self._n_trace[best_index[0]-1]

        self._computation_time[index] = computation_time

    def reached_convergence(self,
                            tolerance: float) -> bool:
        """
        Has the cost trace reached convergence within a tolerance margin ?

        :param tolerance: the allowed root-mean-square deviation.

        :return: Whether the cost trace has converged.
        """
        if self._iteration_counter < self._window_size or not self._detect_convergence:
            return False

        # in case of rescue, if all workers end up on the same position, consider it as convergence
        if np.any(self._rescued[self._iteration_counter - 1]):
            if np.all(np.all(self._position_trace[self.position_counter - 1] ==
                             self._position_trace[self.position_counter - 1][0], axis=1)):
                return True

        position_slice = slice(self.position_counter - self._window_size, self.position_counter)

        mean_window = np.mean(self._best_position_trace[position_slice, -3])
        RMSD = np.sqrt(np.sum(
            (self._best_position_trace[position_slice, -3] - mean_window) ** 2
        ) / (self._window_size - 1))

        return RMSD < tolerance

    def mean_acceptance_fraction(self) -> float:
        """
        Get the mean proportion of accepted proposition in the last <window_size> propositions over all walkers.

        :return: The mean proportion of accepted proposition in the last <window_size> propositions over all walkers.
        """
        if self._iteration_counter < self._window_size:
            return np.nan

        return float(np.mean(
            [np.sum(self._accepted[self._iteration_counter - self._window_size:self._iteration_counter, w]) /
             self._window_size for w in range(self.nb_walkers)]))

    def are_stuck(self) -> list[bool]:
        """
        Detect which walkers are stuck at the same position within the last <window_size> positions.

        :return: The list of stuck walkers.
        """
        if self._iteration_counter < self._window_size:
            return [False for _ in range(self.nb_walkers)]

        return [np.sum(self._accepted[self._iteration_counter - self._window_size:self._iteration_counter, w]) == 0
                for w in range(self.nb_walkers)]

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the vector to optimize.

        :return: The number of dimensions of the vector to optimize.
        """
        return self._position_trace.shape[2]

    @property
    def nb_walkers(self) -> int:
        """
        Get the number of parallel walkers.

        :return: The number of parallel walkers.
        """
        return self._position_trace.shape[1]

    @property
    def nb_positions(self) -> int:
        """
        Get the number of saved positions.

        :return: The number of saved positions.
        """
        return self._position_trace.shape[0]

    @property
    def nb_iterations(self) -> int:
        """
        Get the number of elapsed iterations.

        :return: The number of elapsed iterations.
        """
        return self._temperature_trace.shape[0]

    @property
    def position_counter(self) -> int:
        """
        Get the position pointer for saving new positions.

        :return: The position pointer.
        """
        return self._iteration_counter + 1 * self._initialized

    def get_best(self) -> tuple[np.ndarray, float, list[int]]:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        :return: the best vector, the best cost and iteration number that reached it.
        """
        START = 0 + 1 * self._initialized if self.position_counter > 1 else 0

        lookup_array = self._cost_trace[
                       max(START, self.position_counter - self._window_size):self.position_counter].copy()

        # normalize lookup array to account for variance drop
        current_n = self._n_trace[self._iteration_counter-1]
        with np.errstate(divide='ignore'):
            correction_factors = current_n / self._cost_trace_n[max(START, self.position_counter -
                                                                     self._window_size):self.position_counter]

        lookup_array *= correction_factors

        _best_index = list(np.unravel_index(np.argmin(lookup_array), lookup_array.shape))

        _best_index[0] += max(START, self.position_counter - self._window_size)
        corrected_index = _best_index[0]

        while self._cost_trace[corrected_index, _best_index[1]] == self._cost_trace[_best_index[0], _best_index[1]]:
            corrected_index -= 1
            if corrected_index < 0:
                break

        _best_index[0] = corrected_index + 1 * self._initialized

        return self._position_trace[_best_index[0], _best_index[1]], \
            self._cost_trace[_best_index[0], _best_index[1]], \
            _best_index

    def get_position_trace(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get traces for vector, cost values, n at cost evaluation and best vector along iterations.

        :return: Traces for vector, cost values, n at cost evaluation and best vector along iterations.
        """
        return self._position_trace, self._cost_trace, self._cost_trace_n, self._best_position_trace

    def get_parameters_trace(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get traces related to parameters values along iterations.

        :return: Traces related to parameters values along iterations.
        """
        return self._temperature_trace, self._n_trace, self._sigma_trace

    def get_acceptance_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get traces related to acceptance rates along iterations.

        :return: Traces related to acceptance rates along iterations.
        """
        return self._accepted, self._rescued

    def plot_positions(self,
                       save: Optional[Path] = None,
                       true_values: Optional[Sequence[float]] = None,
                       extended: bool = False,
                       show: bool = True) -> None:
        """
        Plot reached positions and costs for the vector to optimize along iterations.

        Args:
            save: optional path to save the plot as an html file.
            true_values: an optional sequence of known true values for each dimension of the vector to optimize.
            extended: plot additional plots ? (mostly for debugging)
            show: render the plot ? (default True)
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
                            vertical_spacing=0)  # min(0.05, 1 / (self.ndim + supp_plots - 1))

        for w in range(self.nb_walkers):
            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=self._cost_trace[:, w],
                                     name=f'Walker #{w}',
                                     marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                     hovertext=[f"<b>Walker</b>: {w}<br>"
                                                f"<b>Cost</b>: {cost:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, cost in enumerate(self._cost_trace[:, w])],
                                     hoverinfo="text",
                                     showlegend=True,
                                     legendgroup=f'Walker #{w}'), row=1, col=1)

            if extended:
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self._cost_trace_n[:, w],
                                         name=f'Walker #{w}',
                                         marker=dict(color='rgba(0, 0, 200, 0.3)'),
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>n at cost evaluation</b>: {cost_n}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost_n in enumerate(self._cost_trace_n[:, w])],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=2, col=1)

        if extended:
            with np.errstate(divide='ignore', invalid='ignore'):
                best_costs = [self._best_position_trace[it, -3] * np.insert(self._n_trace, 0, 0)[it] /
                              self._best_position_trace[it, -1]
                              for it in range(self.nb_positions)]
            best_walkers = [int(self._best_position_trace[it, -2]) for it in range(self.nb_positions)]

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
                                         y=self._position_trace[:, w, d],
                                         marker=dict(color='rgba(0, 0, 0, 0.3)'),
                                         name=f'Walker #{w}',
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Position</b>: {self._position_trace[iteration, w, d]:.4f}<br>"
                                                    f"<b>Cost</b>: {cost:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost in enumerate(self._cost_trace[:, w])],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=d + supp_plots + 1, col=1)

                # add rescue points
                rescue_iterations = np.where(self._rescued[:, w])[0]
                fig.add_trace(go.Scatter(x=rescue_iterations,
                                         y=self._position_trace[rescue_iterations, w, d],
                                         mode='markers',
                                         marker=dict(color='rgba(0, 255, 0, 0.3)',
                                                     symbol=2,
                                                     size=10),
                                         name=f'Rescues for walker #{w}',
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Position</b>: {self._position_trace[iteration, w, d]:.4f}<br>"
                                                    f"<b>Cost</b>: {self._cost_trace[iteration, w]:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration in rescue_iterations],
                                         hoverinfo="text",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}'), row=d + supp_plots + 1, col=1)

            # add best points
            fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                     y=self._best_position_trace[:, d],
                                     mode='markers',
                                     marker=dict(color='rgba(252, 196, 25, 1.)',
                                                 symbol=0,
                                                 size=3),
                                     name='Best cost',
                                     hovertext=[f"<b>Walker</b>: {int(self._best_position_trace[iteration, -2])}<br>"
                                                f"<b>Position</b>: {position:.4f}<br>"
                                                f"<b>Cost</b>: {self._best_position_trace[iteration, -3]:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration}"
                                                for iteration, position in enumerate(self._best_position_trace[:, d])],
                                     hoverinfo="text",
                                     showlegend=d == 0,
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
                    y=np.max(self._position_trace[:, :, d]),
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
                        save: Optional[Path] = None,
                        extended: bool = False,
                        show: bool = True) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.

        Args:
            save: optional path to save the plot as a html file.
            extended: plot additional plots ? (mostly for debugging)
            show: render the plot ? (default True)
        """
        sub_plots = 5 if extended else 4
        titles = ["Temperature", "sigma", "n", "Acceptance fraction (%)"]
        if extended:
            titles.append("Computation time (s)")

        fig = make_subplots(rows=sub_plots, cols=1, shared_xaxes=True,
                            subplot_titles=titles,
                            vertical_spacing=min(0.05, 1 / (sub_plots - 1)))

        if self.nb_iterations:
            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=self._temperature_trace,
                                     name='T',
                                     hovertext=[f"<b>Temperature</b>: {_T:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration+1}"
                                                for iteration, _T in enumerate(self._temperature_trace)],
                                     hoverinfo="text",
                                     showlegend=False), row=1, col=1)

            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=self._sigma_trace,
                                     name='sigma',
                                     hovertext=[f"<b>Sigma</b>: {_sigma:.4f}<br>"
                                                f"<b>Iteration</b>: {iteration+1}"
                                                for iteration, _sigma in enumerate(self._sigma_trace)],
                                     hoverinfo="text",
                                     showlegend=False), row=2, col=1)

            fig.add_trace(go.Scatter(x=list(range(1, self.position_counter + 1)),
                                     y=self._n_trace,
                                     name='n',
                                     hovertext=[f"<b>Number evaluations</b>: {_n}<br>"
                                                f"<b>Iteration</b>: {iteration+1}"
                                                for iteration, _n in enumerate(self._n_trace)],
                                     hoverinfo="text",
                                     showlegend=False), row=3, col=1)

            accepted_proportions = np.zeros((self.nb_walkers, self.position_counter))

            for w in range(self.nb_walkers):
                accepted_proportions[w] = np.concatenate((np.array([np.nan for _ in range(self._window_size)]),
                                                          np.convolve(self._accepted[:, w],
                                                                      np.ones(self._window_size) / self._window_size,
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
                                         y=self._computation_time,
                                         name='T',
                                         hovertext=[f"<b>Time</b>: {_time:.4f}<br>"
                                                    f"<b>Iteration</b>: {iteration + 1}"
                                                    for iteration, _time in enumerate(self._computation_time)],
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
