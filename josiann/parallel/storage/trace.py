# coding: utf-8
# Created on 16/06/2022 18:41
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

from josiann.name_utils import ShapeError
from josiann.storage import Trace


# ====================================================
# code
class ParallelTrace(Trace):

    def __init__(self,
                 nb_iterations: int,
                 shape: tuple[int, int],
                 window_size: int,
                 bounds: np.ndarray,
                 detect_convergence: bool,
                 max_measures: int,
                 convergence_tolerance: float):
        """
        Args:
            nb_iterations:
            shape:
            window_size:
            bounds:
            detect_convergence:
            max_measures:
            convergence_tolerance: the allowed root-mean-square deviation for convergence detection.
        """
        super().__init__(nb_iterations, shape, window_size, bounds, detect_convergence)

        self._max_measures = max_measures
        self._convergence_tolerance = convergence_tolerance

        self._best_position_trace = np.zeros((nb_iterations, self.nb_walkers, self.nb_dimensions + 2), dtype=np.float32)
        self._converged = np.array([False for _ in range(self.nb_walkers)])

    def __repr__(self) -> str:
        return f"Parallel trace of {self._iteration_counter} iteration(s), {self.nb_walkers} parallel problem(s) and " \
               f"{self.nb_dimensions} dimension(s)."

    @property
    def converged(self) -> np.ndarray:
        return self._converged

    def initialize(self,
                   position: np.ndarray,
                   costs: list[float]) -> None:
        """
        Save state zero before running the SA algorithm. This function should be called before actually storing run
            values.

        Args:
            position: the initial vector.
            costs: the initial costs.
        """
        super().initialize(position, costs)

        self._best_position_trace = np.concatenate(([np.hstack((position.copy(),
                                                               np.array([costs]).T,
                                                               np.tile(1, (self.nb_walkers, 1))))],
                                                    self._best_position_trace))

    def update(self,
               index: int,
               best_position: np.ndarray,
               best_cost: np.ndarray,
               best_index: np.ndarray,
               computation_time: float) -> None:
        """
        Store new information.

        Args:
            index: index at which to update the data.
            best_position: best position vector reached since the start of the SA algorithm.
            best_cost: cost associated to the best position.
            best_index: iteration associated to the best position.
            computation_time: time it took to compute this iteration.
        """
        position_index = index + 1 * self._initialized

        self._best_position_trace[position_index, :, :self.nb_dimensions] = best_position
        self._best_position_trace[position_index, :, self.nb_dimensions] = best_cost
        self._best_position_trace[position_index, :, self.nb_dimensions + 1] = self._n_trace[best_index - 1]

        self._computation_time[index] = computation_time

        self._update_convergence()

    def finalize(self) -> None:
        """
        Cleanup traces at the end of the SA algorithm.
        """
        super().finalize()

        self._best_position_trace = self._best_position_trace[:self.position_counter]

    def _update_convergence(self) -> None:
        """
        For each parallel problem, has the cost trace reached convergence within a tolerance margin ?
        """
        if self._iteration_counter < self._window_size or not self._detect_convergence:
            return

        position_slice = slice(self.position_counter - self._window_size, self.position_counter)

        mean_window = np.mean(self._best_position_trace[position_slice, :, -2], axis=0)
        RMSD = np.sqrt(
            np.sum((self._best_position_trace[position_slice, :, -2] - mean_window) ** 2, axis=0) /
            (self._window_size - 1)
        )

        self._converged |= (RMSD < self._convergence_tolerance)

    def get_best(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        Returns:
            The best vector, the best cost and iteration number that reached it.
        """
        START = 0 + 1 * self._initialized if self.position_counter > 1 else 0

        lookup_array = self._cost_trace[
                       max(START, self.position_counter - self._window_size):self.position_counter].copy()

        start_lookup_index = np.array([max(START, self.position_counter - self._window_size)
                                       for _ in range(self.nb_walkers)])

        # correct lookup array for walkers that already converged
        for walker_index in range(self.nb_walkers):
            if self._converged[walker_index]:
                where_is_nan = np.isnan(self._cost_trace[:, walker_index])
                if np.any(where_is_nan):
                    last_valid_index = np.argmax(where_is_nan)
                    lookup_array[:, walker_index] = np.nan
                    lookup_array[:, walker_index] = \
                        self._cost_trace[max(START, last_valid_index - self._window_size):last_valid_index, walker_index]
                    start_lookup_index[walker_index] = max(START, last_valid_index - self._window_size)

        # normalize lookup array to account for variance drop
        current_n = self._n_trace[self._iteration_counter - 1]
        with np.errstate(divide='ignore'):
            correction_factors = current_n / self._cost_trace_n[max(START, self.position_counter -
                                                                    self._window_size):self.position_counter]

        lookup_array *= correction_factors

        _best_indices = np.nanargmin(lookup_array, axis=0) + start_lookup_index
        
        for i in range(self.nb_walkers):
            corrected_index = _best_indices[i]
    
            while self._cost_trace[corrected_index, i] == self._cost_trace[_best_indices[i], i]:
                corrected_index -= 1
                if corrected_index < 0:
                    break

            _best_indices[i] = corrected_index + 1 * self._initialized

        return self._position_trace[_best_indices, np.arange(self.nb_walkers)], \
            self._cost_trace[_best_indices, np.arange(self.nb_walkers)], \
            _best_indices

    def get_acceptance_trace(self) -> np.ndarray:
        """
        Get traces related to acceptance rates along iterations.

        :return: Traces related to acceptance rates along iterations.
        """
        return self._accepted

    def plot_positions(self,
                       save: Path | None = None,
                       true_values: np.ndarray | None = None,
                       extended: bool = False,
                       show: bool = True) -> None:
        """
        Plot reached positions and costs for the vector to optimize along iterations.

        Args:
            save: optional path to save the plot as a html file.
            true_values: an optional sequence of known true values for each dimension of the vector to optimize.
            extended: plot additional plots ? (mostly for debugging)
            show: render the plot ? (default True)
        """
        if true_values is not None:
            if true_values.ndim != 2:
                raise ShapeError('The vector of true values should have 2 dimensions.')
            if true_values.shape != (self.nb_walkers, self.nb_dimensions):
                raise ShapeError(f'The vector of true values should have {self.nb_walkers}x{self.nb_dimensions} '
                                 f'values.')

        supp_plots = 3 if extended else 1
        titles = ["Costs"] + ['' for _ in range(self.nb_walkers-1)]
        if extended:
            titles += ["n at cost evaluation"] + ['' for _ in range(self.nb_walkers-1)]
            titles += ["Best cost evolution"] + ['' for _ in range(self.nb_walkers-1)]
        for i in range(self.nb_dimensions):
            titles += [f'Dimension {i}'] + ['' for _ in range(self.nb_walkers-1)]

        fig = make_subplots(rows=self.nb_dimensions + supp_plots, cols=self.nb_walkers,
                            shared_xaxes='all',
                            subplot_titles=titles,
                            vertical_spacing=0.5 / (1.5 * (self.nb_dimensions + supp_plots) - 0.5),
                            horizontal_spacing=0.1)

        for w in range(self.nb_walkers):
            # -----------------------------------------------------------------
            # Cost trace
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
                                     legendgroup=f'Walker #{w}'), row=1, col=w+1)

        if extended:
            for w in range(self.nb_walkers):
                # -----------------------------------------------------------------
                # <n> at cost evaluation
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
                                         legendgroup=f'Walker #{w}'), row=2, col=w+1)

                # -----------------------------------------------------------------
                # Best cost
                with np.errstate(divide='ignore', invalid='ignore'):
                    best_costs = [self._best_position_trace[it, w, -2] * np.insert(self._n_trace, 0, 0)[it] /
                                  self._best_position_trace[it, w, -1]
                                  for it in range(self.nb_positions)]

                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=best_costs,
                                         name='Best cost evolution',
                                         marker=dict(color='rgba(252, 196, 25, 1.)'),
                                         hovertext=[f"<b>Walker</b>: {w}<br>"
                                                    f"<b>Cost</b>: {cost}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost in enumerate(best_costs)],
                                         hoverinfo="text",
                                         showlegend=False), row=3, col=w+1)

        # -----------------------------------------------------------------
        # Dimensions
        for d in range(self.nb_dimensions):
            for w in range(self.nb_walkers):
                # add exploration trace
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self._explored[:, w, d],
                                         mode='markers',
                                         marker=dict(color=[f'rgba(34, 92, 150, '
                                                            f'{self._cost_trace_n[iteration, w] / self._max_measures})'
                                                            for iteration in range(len(self._explored))],
                                                     symbol=1,
                                                     size=10),
                                         name='Exploration',
                                         hovertext=[f"<b>Position</b>: {self._position_trace[iteration, w, d]:.4f}<br>"
                                                    f"<b>Cost</b>: {cost:.4f}<br>"
                                                    f"<b>Evaluations</b>: {self._cost_trace_n[iteration, w]}<br>"
                                                    f"<b>Iteration</b>: {iteration}"
                                                    for iteration, cost in enumerate(self._cost_trace[:, w])],
                                         hoverinfo="text",
                                         showlegend=d == 0,
                                         legendgroup='Exploration'),
                              row=d + supp_plots + 1, col=w + 1)

                # add position trace
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self._position_trace[:, w, d],
                                         marker=dict(color='rgba(0, 0, 0, 0.3)'),
                                         name=f'Walker #{w}',
                                         hoverinfo="skip",
                                         showlegend=False,
                                         legendgroup=f'Walker #{w}',
                                         xaxis=f'x{w + 1}',
                                         yaxis=f'y{d + supp_plots + 1}'),
                              row=d + supp_plots + 1, col=w+1)

                # add best points
                fig.add_trace(go.Scatter(x=list(range(self.nb_positions)),
                                         y=self._best_position_trace[:, w, d],
                                         mode='markers',
                                         marker=dict(color='rgba(252, 196, 25, 1.)',
                                                     symbol=0,
                                                     size=3),
                                         name='Best cost',
                                         hovertext=[
                                             f"<b>Position</b>: {position:.4f}<br>"
                                             f"<b>Cost</b>: {self._best_position_trace[iteration, w, -2]:.4f}<br>"
                                             f"<b>Iteration</b>: {iteration}"
                                             for iteration, position in enumerate(self._best_position_trace[:, w, d])
                                         ],
                                         hoverinfo="text",
                                         showlegend=d == 0,
                                         legendgroup='Best cost'),
                              row=d + supp_plots + 1, col=w+1)

                if true_values is not None:
                    fig.add_trace(go.Scatter(x=[0, self.nb_positions - 1],
                                             y=[true_values[w, d], true_values[w, d]],
                                             mode='lines',
                                             marker=dict(color='rgba(200, 0, 0, 1)'),
                                             line=dict(dash='dash'),
                                             name='True value',
                                             showlegend=False),
                                  row=d + supp_plots + 1, col=w+1)

                    fig.add_annotation(
                        x=self.nb_positions,
                        y=true_values[w, d],
                        xref=f"x1",
                        yref=f"y1",
                        text=f"True value : {true_values[w, d]}",
                        borderwidth=0,
                        borderpad=4,
                        bgcolor="#eb9a9a",
                        opacity=0.8,
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=0,
                        row=d + supp_plots + 1,
                        col=w + 1
                    )

        for i in range(self.nb_dimensions + 1):
            fig.layout.annotations[i].update(x=0.025, xanchor='left')

        fig['layout'].update(height=400 * (self.nb_dimensions + 2),
                             width=400 * self.nb_walkers,
                             margin=dict(t=40, b=10, l=10, r=10),
                             xaxis_range=[0, self.nb_positions - 1], template='plotly_white')

        if show:
            fig.show()

        if save is not None:
            fig.write_html(str(save))
