# Created on 02/12/2022 23:47
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from josiann.storage.parallel.parameters import ParallelBaseParameters
    from josiann.storage.parameters import BaseParameters


# ====================================================
# code
# parameters computation --------------------------------------------------------------------------
def acceptance_log_probability(current_cost: float, new_cost: float, temperature: float) -> float:
    """
    Compute the acceptance probability for a new proposed cost, given the current cost and a temperature.

    Args:
        current_cost: the current cost.
        new_cost: the new proposed cost.
        temperature: the current temperature.

    Returns:
        The probability of acceptance of the new proposed cost.
    """
    return (current_cost - new_cost) / temperature


def sigma(k: int, T_0: float, alpha: float, epsilon: float) -> float:
    """
    Compute the estimated standard deviation at iteration k.

    Args:
        k: the iteration number.
        T_0: initial temperature value.
        alpha: rate of temperature decrease.
        epsilon: parameter in (0, 1) for controlling the rate of standard deviation decrease
            (bigger values yield steeper descent profiles)

    Returns:
        The estimated standard deviation.
    """
    return T_0 * (alpha * (1 - epsilon)) ** k


def n(k: int, parameters: BaseParameters | ParallelBaseParameters) -> int:
    """
    Compute the number of necessary measures at iteration k.

    Args:
        k: the iteration number.
        parameters: needed parameters.

    Returns:
        The number of necessary measures.
    """
    if parameters.max_measures == 1:
        return 1

    return int(
        np.ceil(
            (parameters.max_measures * parameters.sigma_max**2)
            / (
                (parameters.max_measures - 1) * sigma(k, parameters.T_0, parameters.alpha, parameters.epsilon) ** 2
                + parameters.sigma_max**2
            )
        )
    )


def T(k: int, T_0: float, alpha: float) -> float:
    """
    Compute the temperature at iteration k.

    Args:
        k: the iteration number.
        T_0: initial temperature value.
        alpha: rate of temperature decrease.

    Returns:
        The temperature.
    """
    return T_0 * alpha**k


def updated_mean(last_n: int, last_mean: float, new_values: npt.NDArray[np.floating[Any]]) -> float:
    """
    Compute a new mean by integrating new values :
        last_mean is transformed back to sum > new_values are added > the sum is divided by the new total number of
        elements
    """
    return (last_mean * last_n + float(np.sum(new_values))) / (last_n + len(new_values))
