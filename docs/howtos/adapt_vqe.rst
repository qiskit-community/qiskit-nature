Find ground state energy using AdaptVQE
=======================================

This guide outlines how the :class:`~qiskit_algorithms.AdaptVQE` algorithm can
be used to find the ground state solutions of natural science problems.


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

4. We setup a :class:`~qiskit_algorithms.VQE`:

.. testcode::

    import numpy as np
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit.primitives import Estimator
    vqe = VQE(Estimator(), ansatz, SLSQP())
    vqe.initial_point = np.zeros(ansatz.num_parameters)

5. We setup the :class:`~qiskit_algorithms.AdaptVQE`:

.. testcode::

    from qiskit_algorithms import AdaptVQE
    adapt_vqe = AdaptVQE(vqe)
    adapt_vqe.supports_aux_operators = lambda: True  # temporary fix

6. We wrap everything in a :class:`~qiskit_nature.second_q.algorithms.GroundStateEigensolver`:

.. testcode::

    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    solver = GroundStateEigensolver(mapper, adapt_vqe)

7. We solve the problem:

.. testcode::

    result = solver.solve(problem)

    print(f"Total ground state energy = {result.total_energies[0]:.4f}")

.. testoutput::

    Total ground state energy = -1.1373
