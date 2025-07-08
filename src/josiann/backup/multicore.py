# Created on 13/01/2023 09:28
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from multiprocessing.managers import BaseManager, NamespaceProxy  # type: ignore[attr-defined]

import numpy as np
import numpy.typing as npt

from josiann.backup.backup import EVALUATION, SequentialBackup


# ====================================================
# code
class BackupManager(BaseManager):
    """
    Manager for passing the Backup objects during multiprocessing.
    """


class BackupProxy(NamespaceProxy):  # type: ignore[misc]
    """
    Proxy for accessing methods of the Backup objects during multiprocessing.
    """

    _exposed_ = (
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "save",
        "get_previous_evaluations",
    )

    def save(self, position: npt.NDArray[np.float64 | np.int64], evaluation: EVALUATION) -> None:
        """
        Proxy for Backup.save() method.
        """
        callmethod = object.__getattribute__(self, "_callmethod")
        return callmethod(self.save.__name__, (position, evaluation))  # type: ignore[no-any-return]

    def get_previous_evaluations(self, position: npt.NDArray[np.float64 | np.int64]) -> EVALUATION:
        """
        Proxy for Backup.get_previous_evaluations() method.
        """
        callmethod = object.__getattribute__(self, "_callmethod")
        return callmethod(self.get_previous_evaluations.__name__, (position,))  # type: ignore[no-any-return]


BackupManager.register("SequentialBackup", SequentialBackup, BackupProxy)
