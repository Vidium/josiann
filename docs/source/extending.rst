Extending Josiann
=================

Custom move functions can be used with Josiann as long as they inherit from one of the base `Move` classes :

- :class:`josiann.moves.base.Move` base class for defining any move.
- :class:`josiann.moves.discrete.DiscreteMove` base class for defining discrete moves.
- :class:`josiann.moves.ensemble.EnsembleMove` base class for defining moves with multiple walkers.
- :class:`josiann.parallel.moves.base.ParallelMove` base class for defining moves for the parallel mode of *Josiann*.

Minimal requirement
-------------------

Custom moves must at least implement the `_get_proposal()` method, which generates candidate positions for iteration
:math:`k+1` by altering the position vector of iteration :math:`k`.
Function `_get_proposal()` has the following signature :

.. code-block:: python

    def _get_proposal(self,
              x: npt.NDArray[Union[np.int64, np.float64]],
              state: State) -> npt.NDArray[Union[np.int64, np.float64]]:
        """
        Generate a new proposed vector x.

        Args:
            x: current vector x of shape (ndim,).
            state: current state of the SA algorithm.

        Returns:
            New proposed vector x of shape (ndim,).
        """
        ...


The `state` parameter is a :class:`josiann.moves.base.State` object holding the current state of the SA algorithm.
