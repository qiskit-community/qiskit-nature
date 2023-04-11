.. _how-to-vqe-ucc:

Use a UCC-like ansatz with a VQE
================================

When using a :class:`~qiskit_nature.second_q.circuit.library.UCC`-style ansatz with a
:class:`~qiskit.algorithms.minimum_eigensolvers.VQE` one needs to pay particular attention to the
:attr:`~qiskit.algorithms.minimum_eigensolvers.VQE.initial_point` attribute which indicates from
which set of initial parameters the optimization routine should start.
By default, VQE will start from a *random* initial point. In this how to we show how one
can set a custom initial point instead (for example to guarantee that one starts from the
Hartree-Fock state).

1. We obtain an :class:`~qiskit_nature.second_q.problems.ElectronicStructureProblem`
   which we want to solve:

.. testcode::

    from qiskit_nature.second_q.drivers import PySCFDriver
    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
    problem = driver.run()

2. We setup our :class:`~qiskit_nature.second_q.mappers.QubitMapper`:

.. testcode::

    from qiskit_nature.second_q.mappers import JordanWignerMapper
    mapper = JordanWignerMapper()

3. We setup our ansatz:

.. testcode::

    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )

4. We setup a :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`:

.. testcode::

    import numpy as np
    from qiskit.algorithms.optimizers import SLSQP
    from qiskit.algorithms.minimum_eigensolvers import VQE
    from qiskit.primitives import Estimator
    vqe = VQE(Estimator(), ansatz, SLSQP())

5. Now comes the key step: choosing the initial point. Since we picked the
   :class:`~qiskit_nature.second_q.circuit.library.HartreeFock` initial
   state before, in order to ensure we start from that, we need to initialize our
   ``initial_point`` with all-zero parameters. One way to do that is like so:

.. testcode::

    vqe.initial_point = np.zeros(ansatz.num_parameters)

Alternatively, one can also use
:class:`~qiskit_nature.second_q.algorithms.initial_points.HFInitialPoint` like so:

.. testcode::

    from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
    initial_point = HFInitialPoint()
    initial_point.ansatz = ansatz
    initial_point.problem = problem
    vqe.initial_point = initial_point.to_numpy_array()

This may seem like it is not adding a lot of benefit, but the key aspect here is that you can build
your code on top of the :class:`~qiskit_nature.second_q.algorithms.initial_points.InitialPoint`
interface based on which we also have the
:class:`~qiskit_nature.second_q.algorithms.initial_points.MP2InitialPoint` which allows you to start
from an MP2 starting point like so:

.. testcode::

    from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
    initial_point = MP2InitialPoint()
    initial_point.ansatz = ansatz
    initial_point.problem = problem
    vqe.initial_point = initial_point.to_numpy_array()

6. Finally, we can now actually solve our problem:

.. testcode::

    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    solver = GroundStateEigensolver(mapper, vqe)
    result = solver.solve(problem)

    print(f"Total ground state energy = {result.total_energies[0]:.4f}")

.. testoutput::

    Total ground state energy = -1.1373
