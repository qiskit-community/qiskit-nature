.. _how-to-vqe-uvcc:

Use a UVCC-like ansatz with a VQE
=================================

When using a :class:`~qiskit_nature.second_q.circuit.library.UVCC`-style ansatz with a
:class:`~qiskit.algorithms.minimum_eigensolvers.VQE` one needs to pay particular attention to the
:attr:`~qiskit.algorithms.minimum_eigensolvers.VQE.initial_point` attribute which indicates from
which set of initial parameters the optimization routine should start.
By default, VQE will start from a *random* initial point. In this how to we show how one
can set a custom initial point instead (for example to guarantee that one starts from the
VSCF state).

The basics of this how-to are identical to the UCC-like ansatz how-to (TODO: add link). Thus, here
we will simply show how to use the
:class:`~qiskit_nature.second_q.algorithms.initial_points.VSCFInitialPoint` like so:

1. Assuming we already have our :class:`~qiskit_nature.second_q.problems.VibrationalStructureProblem`
   and :class:`~qiskit_nature.second_q.mappers.QubitMapper`:

.. testcode::

    from qiskit_nature.second_q.mappers import DirectMapper
    from qiskit_nature.second_q.problems import VibrationalStructureProblem
    problem: VibrationalStructureProblem = ...
    num_modals = [2, 2, 2]  # some example of what problem.num_modals might yield
    mapper = DirectMapper()

2. We setup our ansatz:

.. testcode::

    from qiskit_nature.second_q.circuit.library import UVCCSD, VSCF
    ansatz = UVCCSD(
        num_modals,
        mapper,
        initial_state=VSCF(
            num_modals,
            mapper,
        ),
    )

3. We setup a :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`:

.. testcode::

    import numpy as np
    from qiskit.algorithms.optimizers import SLSQP
    from qiskit.algorithms.minimum_eigensolvers import VQE
    from qiskit.primitives import Estimator
    vqe = VQE(Estimator(), ansatz, SLSQP())

4. Now comes the key step: choosing the initial point. Since we picked the
   :class:`~qiskit_nature.second_q.circuit.library.VSCF` initial state before,
   in order to ensure we start from that, we need to initialize our ``initial_point``
   with all-zero parameters. One way to do that is like so:

.. testcode::

    vqe.initial_point = np.zeros(ansatz.num_parameters)

Alternatively, one can also use
:class:`~qiskit_nature.second_q.algorithms.initial_points.VSCFInitialPoint` like so:

.. testcode::

    from qiskit_nature.second_q.algorithms.initial_points import VSCFInitialPoint
    initial_point = VSCFInitialPoint()
    initial_point.ansatz = ansatz
    initial_point.problem = problem
    vqe.initial_point = initial_point.to_numpy_array()

Just like in the :ref:`UCC-ansatz case <how-to-vqe-ucc>`, this is mostly useful when building more
code on top of the :class:`~qiskit_nature.second_q.algorithms.initial_points.InitialPoint` interface.
