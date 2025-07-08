# Created on 03/02/2022 10:40
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import josiann.typing as jot
from josiann.errors import ShapeError
from josiann.storage.parameters import SAParameters

logger = logging.getLogger(__name__)

PLOTTING_ENABLED = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

except ImportError:
    logger.info("Plotly is not installed, consider installing it or running 'pip install josiann[plot]'.")

else:
    PLOTTING_ENABLED = True


# ====================================================
# code
@dataclass
class Position:
    x: npt.NDArray[np.float64]  # (nb_walkers, nb_dimensions)
    cost: npt.NDArray[np.float64]  # (nb_walkers,)
    n: npt.NDArray[np.float64]  # (nb_walkers,)


class PositionTrace:
    # region magic methods
    def __init__(
        self,
        run_parameters: SAParameters,
        nb_iterations: int,
        nb_walkers: int,
        nb_dimensions: int,
        initial_position: npt.NDArray[np.float64],
        initial_cost: npt.NDArray[np.float64],
    ):
        """
        Args:
            nb_iterations: number of expected iterations for the SA algorithm.
            nb_walkers: number of walkers in parallel.
            nb_dimensions: number of dimensions per problem.
            initial_position: initial positions before running the SA algorithm.
            initial_cost: cost of initial positions.
        """
        self.parameters = run_parameters
        self.nb_walkers = nb_walkers
        self.nb_dimensions = nb_dimensions

        # /!\ store nb_iter + 1 values (initial values + iterations)
        self.position_trace = np.zeros((nb_iterations + 1, nb_walkers, nb_dimensions), dtype=np.float64)
        self.explored_trace = np.zeros((nb_iterations + 1, nb_walkers, nb_dimensions), dtype=np.float64)
        self.best_position_trace = np.zeros((nb_iterations + 1, nb_walkers, nb_dimensions + 2), dtype=np.float64)
        self.cost_trace = np.zeros((nb_iterations + 1, nb_walkers), dtype=np.float64)
        self.cost_trace_n = np.zeros((nb_iterations + 1, nb_walkers), dtype=np.int64)
        self.explored_cost_trace = np.zeros((nb_iterations + 1, nb_walkers), dtype=np.float64)
        self._accepted = np.zeros((nb_iterations + 1, nb_walkers), dtype=bool)
        self.rescued = np.zeros((nb_iterations + 1, nb_walkers), dtype=np.float64)

        self.converged_at_iteration = -1 * np.ones(nb_walkers)

        # store initial values
        self.position_trace[0] = initial_position
        self.explored_trace[0] = initial_position
        self.cost_trace[0] = initial_cost
        self.cost_trace_n[0] = 1
        self.explored_cost_trace[0] = initial_cost
        self._accepted[0] = True

        self._set_best(-1, 1)

    # endregion

    # region attributes
    @property
    def converged(self) -> npt.NDArray[np.bool_]:
        return self.converged_at_iteration > -1

    @property
    def nb_iterations(self) -> int:
        return self.position_trace.shape[0] - 1

    # endregion

    # region methods
    def finalize(self, iteration: int) -> None:
        """
        Cleanup trace at the end of the SA algorithm.
        """
        self.position_trace = self.position_trace[: iteration + 2]
        self.explored_trace = self.explored_trace[: iteration + 2]
        self.best_position_trace = self.best_position_trace[: iteration + 2]
        self.cost_trace = self.cost_trace[: iteration + 2]
        self.cost_trace_n = self.cost_trace_n[: iteration + 2]
        self.explored_cost_trace = self.explored_cost_trace[: iteration + 2]
        self._accepted = self._accepted[: iteration + 2]
        self.rescued = self.rescued[: iteration + 2]

    def store(
        self,
        iteration: int,
        position: npt.NDArray[np.float64 | np.int64],
        costs: npt.NDArray[np.float64],
        current_n: int,
        accepted: npt.NDArray[np.bool_],
        explored: npt.NDArray[np.float64 | np.int64],
        explored_costs: npt.NDArray[np.float64],
    ) -> None:
        """
        Save the current position of the vector to optimize, the current cost, temperature and number of averaged
            function evaluations.

        Args:
            iteration: iteration index for storing the data.
            position: the current positions.
            costs: the current costs.
            current_n: the current number of averaged function evaluations.
            accepted: were the current propositions accepted ?
            explored: the array of explored propositions.
            explored_costs: costs associated to explored positions.

        Returns:
            The index at which was stored the data.
        """
        self.position_trace[iteration + 1] = position
        self.explored_trace[iteration + 1] = explored

        self.cost_trace[iteration + 1] = costs
        self.explored_cost_trace[iteration + 1] = explored_costs

        self.cost_trace_n[iteration + 1, accepted] = current_n
        self.cost_trace_n[iteration + 1, ~accepted] = self.cost_trace_n[iteration, ~accepted]

        self._accepted[iteration + 1] = accepted

        self._set_best(iteration, current_n)
        self._update_convergence(iteration)

    def update(
        self,
        iteration: int,
        position: npt.NDArray[jot.DType],
        costs: npt.NDArray[np.float64],
        last_ns: npt.NDArray[np.int64],
        rescued: npt.NDArray[np.bool_],
    ) -> None:
        """
        Updates positions and costs. Also store new information.

        Args:
            iteration: iteration index for storing the data.
            position: the current vector.
            costs: the current costs.
            last_ns: the number of averaged function evaluations.
            rescued: were the walkers rescued at this iteration ?
        """
        if np.any(rescued):
            self.position_trace[iteration + 1, rescued] = position[rescued]
            self.cost_trace[iteration + 1, rescued] = costs[rescued]

            self.rescued[iteration + 1] = np.array(rescued)

            self.cost_trace_n[iteration + 1] = last_ns

    def _set_best(self, iteration: int, current_n: int) -> None:
        """
        Get the best vector with associated cost and iteration at which it was reached.

        Args:
            iteration: iteration index at which to get the best vector.
            current_n: the current number of averaged function evaluations.

        Returns:
            The best vector, the best cost and iteration number that reached it.
        """
        _START = max(0, iteration - self.parameters.window_size + 2)
        _STOP = iteration + 2

        lookup_array = self.cost_trace[_START:_STOP].copy()

        start_lookup_index = np.repeat(_START, self.nb_walkers)

        # # correct lookup array for walkers that already converged
        # last_valid_indices = np.argmax(np.isnan(self.cost_trace), axis=0) - 1
        #
        # for walker_index in range(self.nb_walkers):
        #     if last_valid_indices[walker_index] != -1:
        #         lookup_array[:, walker_index] = self.cost_trace[last_valid_indices[walker_index], walker_index]
        #         start_lookup_index[walker_index] = last_valid_indices[walker_index]

        # normalize lookup array to account for variance drop
        lookup_array *= current_n / self.cost_trace_n[_START:_STOP]

        # ---------------------------------------------------------------------
        # where walkers have converged, copy previous best position
        self.best_position_trace[iteration + 1, self.converged, : self.nb_dimensions] = self.best_position_trace[
            iteration, self.converged, : self.nb_dimensions
        ]
        self.best_position_trace[iteration + 1, self.converged, self.nb_dimensions] = self.best_position_trace[
            iteration, self.converged, self.nb_dimensions
        ]
        self.best_position_trace[iteration + 1, self.converged, self.nb_dimensions + 1] = self.best_position_trace[
            iteration, self.converged, self.nb_dimensions + 1
        ]

        # ---------------------------------------------------------------------
        # where walkers have not converged, find best position
        best_iteration = np.nanargmin(lookup_array[:, ~self.converged], axis=0) + start_lookup_index[~self.converged]

        # find first occurence of that best iteration (/!\ might be before _START)
        best_iteration = np.argmax(
            self.cost_trace[:_STOP, ~self.converged] == self.cost_trace[best_iteration, ~self.converged],
            axis=0,
        )

        self.best_position_trace[iteration + 1, ~self.converged, : self.nb_dimensions] = self.position_trace[
            best_iteration, ~self.converged
        ]
        self.best_position_trace[iteration + 1, ~self.converged, self.nb_dimensions] = self.cost_trace[
            best_iteration, ~self.converged
        ]
        self.best_position_trace[iteration + 1, ~self.converged, self.nb_dimensions + 1] = self.cost_trace_n[
            best_iteration, ~self.converged
        ]

    def _update_convergence(self, iteration: int) -> None:
        """
        For each parallel problem, has the cost trace reached convergence within a tolerance margin ?

        Args:
            iteration: current iteration number.
        """
        if not self.parameters.base.detect_convergence or iteration < self.parameters.window_size:
            return

        position_slice = slice(iteration - self.parameters.window_size, iteration)

        mean_window = np.mean(self.cost_trace[position_slice], axis=0)
        RMSD = np.sqrt(
            np.sum((self.cost_trace[position_slice] - mean_window) ** 2, axis=0) / (self.parameters.window_size - 1)
        )

        self.converged_at_iteration[(RMSD < self.parameters.base.tol) & (self.converged_at_iteration == -1)] = iteration

    def get_best(self, iteration: int | None = None) -> Position:
        if iteration is None:
            iteration = self.nb_iterations - 1

        return Position(
            self.best_position_trace[iteration + 1, :, : self.nb_dimensions].copy(),
            self.best_position_trace[iteration + 1, :, self.nb_dimensions].copy(),
            self.best_position_trace[iteration + 1, :, self.nb_dimensions + 1].copy(),
        )

    def get_best_all(self, iteration: int | None = None) -> Position:
        best = self.get_best(iteration)
        best_index = np.argmin(best.cost)

        return Position(best.x[best_index], best.cost[best_index], best.n[best_index])

    def mean_acceptance_fraction(self, iteration: int) -> float:
        """
        Get the mean proportion of accepted proposition in the last <window_size> propositions over all walkers.

        Args:
            iteration: current iteration number.

        Returns:
            The mean proportion of accepted proposition in the last <window_size> propositions over all walkers.
        """
        if iteration < self.parameters.window_size:
            return np.nan

        _START = iteration - self.parameters.window_size + 1
        _STOP = iteration + 1

        return float(
            np.mean(
                [np.sum(self._accepted[_START:_STOP, w]) / self.parameters.window_size for w in range(self.nb_walkers)]
            )
        )

    def are_stuck(self, iteration: int) -> npt.NDArray[np.bool_]:
        """
        Detect which walkers are stuck at the same position within the last <window_size> positions.

        Args:
            iteration: current iteration number.

        Returns:
            The list of stuck walkers.
        """
        if iteration < self.parameters.window_size:
            return np.zeros(self.nb_walkers, dtype=bool)

        _START = iteration - self.parameters.window_size + 1
        _STOP = iteration + 1

        return np.array([np.sum(self._accepted[_START:_STOP, w]) == 0 for w in range(self.nb_walkers)])

    # endregion


class ParameterTrace:
    # region magic methods
    def __init__(self, nb_iterations: int):
        """
        Args:
            nb_iterations: number of expected iterations for the SA algorithm.
        """
        self.temperature_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.n_trace = np.zeros(nb_iterations, dtype=np.int32)
        self.sigma_trace = np.zeros(nb_iterations, dtype=np.float32)
        self.computation_time = np.zeros(nb_iterations, dtype=np.float32)

    # endregion

    # region attributes
    @property
    def nb_iterations(self) -> int:
        return self.temperature_trace.shape[0]

    # endregion

    # region methods
    def finalize(self, iteration: int) -> None:
        """
        Cleanup trace at the end of the SA algorithm.
        """
        self.temperature_trace = self.temperature_trace[: iteration + 1]
        self.n_trace = self.n_trace[: iteration + 1]
        self.sigma_trace = self.sigma_trace[: iteration + 1]
        self.computation_time = self.computation_time[: iteration + 1]

    def store(
        self,
        iteration: int,
        temperature: float,
        n: int,
        sigma: float,
        computation_time: float,
    ) -> None:
        """
        Save the current position of the vector to optimize, the current cost, temperature and number of averaged
            function evaluations.

        Args:
            iteration: iteration index for storing the data.
            temperature: the current temperature.
            n: the current number of averaged function evaluations.
            sigma: the current estimated standard deviation.
            computation_time: the time required for computing this iteration.

        Returns:
            The index at which was stored the data.
        """
        self.temperature_trace[iteration] = float(temperature)
        self.n_trace[iteration] = int(n)
        self.sigma_trace[iteration] = float(sigma)
        self.computation_time[iteration] = float(computation_time)

    # endregion


class Trace(ABC):
    """
    Object for storing the trace history of an SA run.
    """

    # region magic methods
    def __init__(
        self,
        nb_iterations: int,
        nb_walkers: int,
        nb_dimensions: int,
        run_parameters: SAParameters,
        initial_position: npt.NDArray[np.float64],
        initial_cost: npt.NDArray[np.float64],
    ):
        """
        Instantiate a Trace.

        Args:
            nb_iterations: number of expected iterations for the SA algorithm.
            nb_walkers: number of walkers that run in parallel.
            nb_dimensions: number of dimensions per optimization problem.
            run_parameters: parameters used for running the SA algorithm.
            initial_position: initial positions before running the SA algorithm.
            initial_cost: cost of initial positions.
        """
        # trace values that will change during SA execution
        self.positions = PositionTrace(
            run_parameters,
            nb_iterations,
            nb_walkers,
            nb_dimensions,
            initial_position,
            initial_cost,
        )
        self.parameters = ParameterTrace(nb_iterations)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    # endregion

    # region attributes
    @property
    def nb_walkers(self) -> int:
        """Number of walkers that run in parallel."""
        return self.positions.nb_walkers

    @property
    def nb_dimensions(self) -> int:
        """Number of dimensions per optimization problem."""
        return self.positions.nb_dimensions

    @property
    def nb_iterations(self) -> int:
        """Number of iterations that the SA algorithm run for."""
        return self.positions.nb_iterations

    # endregion

    # region methods
    def finalize(self, iteration: int) -> None:
        """
        When the SA algorithm terminates, finalize() is called on position and parameter traces to delete rows for
        iterations that where never run.

        Args:
            iteration: final iteration that was computed before termination.

        :meta private:
        """
        self.positions.finalize(iteration)
        self.parameters.finalize(iteration)

    @abstractmethod
    def plot_positions(
        self,
        save: Path | None = None,
        true_values: npt.NDArray[Any] | None = None,
        show: bool = True,
        walker_titles: Sequence[str] | None = None,
        dimension_titles: Sequence[str] | None = None,
    ) -> None:
        """
        Plot reached positions and costs for the vector to optimize along iterations.

        Args:
            save: optional path to save the plot as a html file.
            true_values: an optional sequence of known true values for each dimension of the vector to optimize.
            show: render the plot ?
            walker_titles: an optional list of sub-plot titles, one title per parallel walker.
            dimension_titles: an optional list of sub-plot titles, one title per dimension.
        """
        pass

    def plot_parameters(self, save: Path | str | None = None, show: bool = True) -> None:
        """
        Plot temperature, number of repeats per iteration and number of averaged function evaluations along iterations.

        Args:
            save: optional path to save the plot as a html file.
            show: render the plot ?
        """
        if not PLOTTING_ENABLED:
            raise ImportError("Plotly is not installed.")

        titles = ["Temperature", "sigma", "n", "Computation time (s)"]

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.1,
        )

        if self.nb_iterations:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, self.nb_iterations + 1)),
                    y=self.parameters.temperature_trace,
                    name="T",
                    hovertext=[
                        f"<b>Temperature</b>: {_T:.4f}<br><b>Iteration</b>: {iteration + 1}"
                        for iteration, _T in enumerate(self.parameters.temperature_trace)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, self.nb_iterations + 1)),
                    y=self.parameters.sigma_trace,
                    name="sigma",
                    hovertext=[
                        f"<b>Sigma</b>: {_sigma:.4f}<br><b>Iteration</b>: {iteration + 1}"
                        for iteration, _sigma in enumerate(self.parameters.sigma_trace)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, self.nb_iterations + 1)),
                    y=self.parameters.n_trace,
                    name="n",
                    hovertext=[
                        f"<b>Number evaluations</b>: {_n}<br><b>Iteration</b>: {iteration + 1}"
                        for iteration, _n in enumerate(self.parameters.n_trace)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, self.nb_iterations + 1)),
                    y=self.parameters.computation_time,
                    name="T",
                    hovertext=[
                        f"<b>Time</b>: {_time:.4f}<br><b>Iteration</b>: {iteration + 1}"
                        for iteration, _time in enumerate(self.parameters.computation_time)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

        fig.update_layout(
            height=800,
            width=600,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="#FFF",
            plot_bgcolor="#FFF",
            font_color="#000000",
        )

        for i in range(4):
            fig.layout.annotations[i].update(x=0.025, xanchor="left")

        if show:
            fig.show()

        if save is not None:
            fig.write_html(str(save))

    # endregion


class OneTrace(Trace):
    """
    Object for storing the trace history of an SA run.
    """

    # region magic methods
    def __repr__(self) -> str:
        return (
            f"Trace of {self.nb_iterations} iteration(s), {self.nb_walkers} walker(s) and "
            f"{self.nb_dimensions} dimension(s)."
        )

    # endregion

    # region methods
    def plot_positions(
        self,
        save: Path | None = None,
        true_values: npt.NDArray[Any] | None = None,
        show: bool = True,
        walker_titles: Sequence[str] | None = None,
        dimension_titles: Sequence[str] | None = None,
    ) -> None:
        """
        Plot reached positions and costs for the vector to optimize along iterations.

        Args:
            save: optional path to save the plot as a html file.
            true_values: an optional sequence of known true values for each dimension of the vector to optimize.
            show: render the plot ? (default True)
            walker_titles: an optional list of sub-plot titles, one title per parallel walker. (default None)
            dimension_titles: an optional list of sub-plot titles, one title per dimension. (default None)
        """
        if true_values is not None and len(true_values) != self.nb_dimensions:
            raise ShapeError(
                f"The vector of true values should have {self.nb_dimensions} dimensions, not {len(true_values)}."
            )

        # if subplot_titles is not None:
        #     if len(subplot_titles) != self.nb_dimensions:
        #         raise ShapeError(f'Expected {self.nb_dimensions} sub-plot titles, got {len(subplot_titles)}.')

        titles = ["Costs", "n", "Best cost evolution"]

        if walker_titles is not None:
            # titles += [f'{subplot_titles[i]}' for i in range(self.nb_dimensions)]
            pass

        else:
            titles += [f"Dimension {i}" for i in range(self.nb_dimensions)]

        fig = make_subplots(
            rows=self.nb_dimensions + 3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.05,
        )

        for w in range(self.nb_walkers):
            fig.add_trace(
                go.Scatter(
                    x=list(range(self.nb_iterations)),
                    y=self.positions.cost_trace[:, w],
                    name=f"Walker #{w}",
                    marker=dict(color="rgba(0, 0, 200, 0.3)"),
                    hovertext=[
                        f"<b>Walker</b>: {w}<br><b>Cost</b>: {cost:.4f}<br><b>Iteration</b>: {iteration}"
                        for iteration, cost in enumerate(self.positions.cost_trace[:, w])
                    ],
                    hoverinfo="text",
                    showlegend=True,
                    legendgroup=f"Walker #{w}",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(self.nb_iterations)),
                    y=self.positions.cost_trace_n[:, w],
                    name=f"Walker #{w}",
                    marker=dict(color="rgba(0, 0, 200, 0.3)"),
                    hovertext=[
                        f"<b>Walker</b>: {w}<br><b>n at cost evaluation</b>: {cost_n}<br><b>Iteration</b>: {iteration}"
                        for iteration, cost_n in enumerate(self.positions.cost_trace_n[:, w])
                    ],
                    hoverinfo="text",
                    showlegend=False,
                    legendgroup=f"Walker #{w}",
                ),
                row=2,
                col=1,
            )

            # Best cost
            best_costs = np.zeros(self.nb_iterations + 1)

            with np.errstate(divide="ignore", invalid="ignore"):
                for it in range(self.nb_iterations + 1):
                    best = self.positions.get_best(it - 1)
                    best_costs[it] = best.cost[w] * np.insert(self.parameters.n_trace, 0, 1)[it] / best.n[w]

            fig.add_trace(
                go.Scatter(
                    x=list(range(self.nb_iterations + 1)),
                    y=best_costs,
                    name="Best cost evolution",
                    marker=dict(color="rgba(252, 196, 25, 1.)"),
                    hovertext=[
                        f"<b>Walker</b>: {w}<br><b>Cost</b>: {cost}<br><b>Iteration</b>: {iteration}"
                        for iteration, cost in enumerate(best_costs)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

            for d in range(self.nb_dimensions):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.nb_iterations)),
                        y=self.positions.position_trace[:, w, d],
                        marker=dict(color="rgba(0, 0, 0, 0.3)"),
                        name=f"Walker #{w}",
                        hovertext=[
                            f"<b>Walker</b>: {w}<br>"
                            f"<b>Position</b>: "
                            f"{self.positions.position_trace[iteration, w, d]:.4f}<br>"
                            f"<b>Cost</b>: {cost:.4f}<br>"
                            f"<b>Iteration</b>: {iteration}"
                            for iteration, cost in enumerate(self.positions.cost_trace[:, w])
                        ],
                        hoverinfo="text",
                        showlegend=False,
                        legendgroup=f"Walker #{w}",
                    ),
                    row=d + 4,
                    col=1,
                )

                # add rescue points
                rescue_iterations = np.where(self.positions.rescued[:, w])[0]
                fig.add_trace(
                    go.Scatter(
                        x=rescue_iterations,
                        y=self.positions.position_trace[rescue_iterations, w, d],
                        mode="markers",
                        marker=dict(color="rgba(0, 255, 0, 0.3)", symbol=2, size=10),
                        name=f"Rescues for walker #{w}",
                        hovertext=[
                            f"<b>Walker</b>: {w}<br>"
                            f"<b>Position</b>: "
                            f"{self.positions.position_trace[iteration, w, d]:.4f}<br>"
                            f"<b>Cost</b>: {self.positions.cost_trace[iteration, w]:.4f}<br>"
                            f"<b>Iteration</b>: {iteration}"
                            for iteration in rescue_iterations
                        ],
                        hoverinfo="text",
                        showlegend=False,
                        legendgroup=f"Walker #{w}",
                    ),
                    row=d + 4,
                    col=1,
                )

                # add best points
                # fig.add_trace(go.Scatter(x=list(range(self.nb_iterations)),
                #                          y=self.positions.best_position_trace[:, d],
                #                          mode='markers',
                #                          marker=dict(color='rgba(252, 196, 25, 1.)',
                #                                      symbol=0,
                #                                      size=3),
                #                          name='Best cost',
                #                          hovertext=[
                #                              f"<b>Walker</b>: "
                #                              f"{int(self.positions.best_position_trace[iteration, -2])}<br>"
                #                              f"<b>Position</b>: {position:.4f}<br>"
                #                              f"<b>Cost</b>: "
                #                              f"{self.positions.best_position_trace[iteration, -3]:.4f}<br>"
                #                              f"<b>Iteration</b>: {iteration}"
                #                             for iteration, position in enumerate(self.positions.best_position_trace[:, d])
                #                          ],
                #                          hoverinfo="text",
                #                          showlegend=d == 0,
                #                          legendgroup='Best cost'), row=d + 4, col=1)

                if true_values is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[0, self.nb_iterations],
                            y=[true_values[d], true_values[d]],
                            mode="lines",
                            marker=dict(color="rgba(200, 0, 0, 1)"),
                            line=dict(dash="dash"),
                            name="True value",
                            showlegend=False,
                        ),
                        row=d + 4,
                        col=1,
                    )

                    fig.add_annotation(
                        x=self.nb_iterations - 1,
                        y=np.max(self.positions.position_trace[:, :, d]),
                        xref=f"x{d + 4}",
                        yref=f"y{d + 4}",
                        text=f"True value : {true_values[d]}",
                        showarrow=False,
                        borderwidth=0,
                        borderpad=4,
                        bgcolor="#eb9a9a",
                        opacity=0.8,
                    )

        fig["layout"].update(
            height=200 * (self.nb_dimensions + 2),
            width=600,
            margin=dict(t=40, b=10, l=10, r=10),
            xaxis_range=[0, self.nb_iterations - 1],
            paper_bgcolor="#FFF",
            plot_bgcolor="#FFF",
            font_color="#000000",
        )

        if show:
            fig.show()

        if save is not None:
            fig.write_html(str(save))

    # endregion
