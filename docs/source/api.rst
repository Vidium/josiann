API
===

.. currentmodule:: josiann

Algorithms
----------

Sequential
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: generated

    sa
    vsa
    mcsa


Parallel
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: generated

    parallel.psa
    parallel.ParallelArgument


Moves
-----


.. autosummary::
    :nosignatures:
    :toctree: generated

    moves.base.State
    moves.base.Move

Sequential
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: generated
    :recursive:

    moves.sequential
    moves.discrete
    moves.ensemble


Parallel
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: generated
    :recursive:

    parallel.moves.base.ParallelMove
    parallel.moves.discrete


Outputs
-------

.. autosummary::
    :nosignatures:
    :toctree: generated

    Result
    Trace


Typing
------

.. autosummary::
    :nosignatures:
    :toctree: generated

    typing.DType
    typing.DT_ARR
