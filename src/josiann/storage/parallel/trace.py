from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm.autonotebook import trange

from josiann.errors import ShapeError
from josiann.storage.trace import Trace

logger = logging.getLogger("josiann")

PLOTTING_ENABLED = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

except ImportError:
    logger.info("Plotly is not installed, consider installing it or running 'pip install josiann[plot]'.")

else:
    PLOTTING_ENABLED = True


class ParallelTrace(Trace):
    # region magic methods
    def __repr__(self) -> str:
        return (
            f"Parallel trace of {self.nb_iterations} iteration(s), {self.nb_walkers} parallel problem(s) and "
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
        if not PLOTTING_ENABLED:
            raise ImportError("Plotly is not installed.")

        if true_values is not None:
            if true_values.ndim != 2:
                raise ShapeError("The vector of true values should have 2 dimensions.")
            if true_values.shape != (self.nb_walkers, self.nb_dimensions):
                raise ShapeError(
                    f"The vector of true values should have {self.nb_walkers}x{self.nb_dimensions} values."
                )

        if walker_titles is not None:
            if len(walker_titles) != self.nb_walkers:
                raise ShapeError(f"Expected {self.nb_walkers} paralle_titles, got {len(walker_titles)}.")

        if dimension_titles is not None:
            if len(dimension_titles) != self.nb_dimensions:
                raise ShapeError(f"Expected {self.nb_dimensions} dimension_titles, got {len(dimension_titles)}.")

        titles = (
            ["Costs"]
            + ["" for _ in range(self.nb_walkers - 1)]
            + ["Best cost evolution"]
            + ["" for _ in range(self.nb_walkers - 1)]
        )

        for i in range(self.nb_dimensions):
            dimension_title = dimension_titles[i] if dimension_titles is not None else f"Dimensions #{i}"

            if walker_titles is not None:
                titles += [
                    f"{dimension_title}<br>{walker_titles[w]}<br>(converged : {self.positions.converged[w]})"
                    for w in range(self.nb_walkers)
                ]

            else:
                titles += [
                    f"{dimension_title}<br>(converged : {self.positions.converged[w]})" for w in range(self.nb_walkers)
                ]

        fig = make_subplots(
            rows=self.nb_dimensions + 2,
            cols=self.nb_walkers,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.5 / (1.5 * (self.nb_dimensions + 2) - 0.5),
            horizontal_spacing=0,
        )

        for w in trange(self.nb_walkers, leave=False):
            # -----------------------------------------------------------------
            # Cost trace
            fig.add_trace(
                go.Scatter(
                    x=list(range(self.positions.nb_iterations + 1)),
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
                col=w + 1,
            )

            # -----------------------------------------------------------------
            # Best cost
            best_costs = np.zeros(self.positions.nb_iterations + 1)

            with np.errstate(divide="ignore", invalid="ignore"):
                for it in range(self.positions.nb_iterations + 1):
                    best = self.positions.get_best(it - 1)
                    best_costs[it] = best.cost[w] * np.insert(self.parameters.n_trace, 0, 1)[it] / best.n[w]

            fig.add_trace(
                go.Scatter(
                    x=list(range(self.positions.nb_iterations + 1)),
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
                row=2,
                col=w + 1,
            )

            # -----------------------------------------------------------------
            # Dimensions
            convergence_index = int(self.positions.converged_at_iteration[w]) if self.positions.converged[w] else -1

            for d in range(self.nb_dimensions):
                # add exploration trace (blue)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.positions.nb_iterations + 1))[:convergence_index],
                        y=self.positions.explored_trace[:convergence_index, w, d],
                        mode="markers",
                        marker=dict(
                            color=[
                                f"rgba(34, 92, 150, "
                                f"{np.insert(self.parameters.n_trace, 0, 1)[it] / self.parameters.n_trace[-1]})"
                                for it in range(self.positions.nb_iterations + 1)
                            ],
                            symbol=1,
                            size=10,
                        ),
                        name="Exploration",
                        hovertext=[
                            f"<b>Position</b>: "
                            f"{self.positions.explored_trace[iteration, w, d]:.4f}<br>"
                            f"<b>Cost</b>: {cost:.4f}<br>"
                            f"<b>Evaluations</b>: "
                            f"{np.insert(self.parameters.n_trace, 0, 1)[iteration]}<br>"
                            f"<b>Iteration</b>: {iteration}"
                            for iteration, cost in enumerate(self.positions.explored_cost_trace[:, w])
                        ],
                        hoverinfo="text",
                        showlegend=d == 0,
                        legendgroup="Exploration",
                    ),
                    row=d + 3,
                    col=w + 1,
                )

                # add position trace (grey)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.positions.nb_iterations + 1))[:convergence_index],
                        y=self.positions.position_trace[:convergence_index, w, d],
                        marker=dict(color="rgba(0, 0, 0, 0.3)"),
                        name=f"Walker #{w}",
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=f"Walker #{w}",
                        xaxis=f"x{w + 1}",
                        yaxis=f"y{d + 3}",
                    ),
                    row=d + 3,
                    col=w + 1,
                )

                # add best points (gold)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(self.positions.nb_iterations + 1))[:convergence_index],
                        y=self.positions.best_position_trace[:convergence_index, w, d],
                        mode="markers",
                        marker=dict(color="rgba(252, 196, 25, 1.)", symbol=0, size=3),
                        name="Best cost",
                        hovertext=[
                            f"<b>Position</b>: {position:.4f}<br>"
                            f"<b>Cost</b>: "
                            f"{self.positions.best_position_trace[iteration, w, -2]:.4f}<br>"
                            f"<b>Iteration</b>: {iteration}"
                            for iteration, position in enumerate(self.positions.best_position_trace[:, w, d])
                        ],
                        hoverinfo="text",
                        showlegend=d == 0,
                        legendgroup="Best cost",
                    ),
                    row=d + 3,
                    col=w + 1,
                )

                # add convergence line (red)
                if convergence_index > -1:
                    fig.add_shape(
                        go.layout.Shape(
                            type="line",
                            yref="paper",
                            xref="x",
                            x0=convergence_index + 1,
                            y0=0,
                            x1=convergence_index + 1,
                            y1=np.nanmax(self.positions.explored_trace[:, w, d]),
                            line=dict(color="firebrick", width=3),
                        ),
                        row=d + 3,
                        col=w + 1,
                    )

                # add true values line (dashed red)
                if true_values is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[0, self.positions.nb_iterations],
                            y=[true_values[w, d], true_values[w, d]],
                            mode="lines",
                            marker=dict(color="rgba(200, 0, 0, 1)"),
                            line=dict(dash="dash"),
                            name="True value",
                            showlegend=False,
                        ),
                        row=d + 3,
                        col=w + 1,
                    )

                    fig.add_annotation(
                        x=self.positions.nb_iterations + 1,
                        y=true_values[w, d],
                        xref="x1",
                        yref="y1",
                        text=f"True value : {true_values[w, d]}",
                        borderwidth=0,
                        borderpad=4,
                        bgcolor="#eb9a9a",
                        opacity=0.8,
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=0,
                        row=d + 3,
                        col=w + 1,
                    )

        # for i in range(self.nb_dimensions + 1):
        #     fig.layout.annotations[i].update(x=0.025, xanchor='left')

        fig["layout"].update(
            height=400 * (self.nb_dimensions + 2),
            width=400 * self.nb_walkers,
            margin=dict(t=40, b=10, l=10, r=10),
            xaxis_range=[0, self.positions.nb_iterations],
            template="plotly_white",
            showlegend=False,
        )

        if show:
            fig.show()

        if save is not None:
            fig.write_html(str(save))

    # endregion
