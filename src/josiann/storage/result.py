# Created on 03/02/2022 11:54
# Author : matteo

# ====================================================
# imports
import numpy as np
import numpy.typing as npt
from attrs import frozen

from josiann.storage.parameters import SAParameters
from josiann.storage.trace import Trace


# ====================================================
# code
@frozen(repr=False)
class Result:
    """
    Object for storing the results of a run.

    Args:
        message: the exit message.
        success: boolean indicating if the SA algorithm did converge to a solution.
        trace: a Trace object with the history of the run.
        parameters: parameters used to run the SA algorithm.
    """

    message: str  #: the exit message.
    success: bool  #: boolean indicating if the SA algorithm did converge to a solution.
    trace: Trace  #: a Trace object with the history of the run.
    parameters: SAParameters  #: parameters used to run the SA algorithm.

    # region magic methods
    def __repr__(self) -> str:
        return (
            f"Result(\n"
            f"\tmessage: {self.message}\n"
            f"\tsuccess: {self.success}\n"
            f"\ttrace: {self.trace}\n"
            f"\tbest: {self.best}"
            f")"
        )

    # endregion

    # region attrbiutes
    @property
    def best(self) -> npt.NDArray[np.float64]:
        """Get the best position vector (with the lowest cost, among all walkers)."""
        return self.trace.positions.get_best_all().x

    # endregion
