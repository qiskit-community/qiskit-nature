Find ground state energy using AdaptVQE
=======================================

As of Qiskit Nature v0.5, the :class:`~qiskit.algorithms.minimum_eigensolvers.AdaptVQE`
algorithm has been migrated to Qiskit Terra (released in v0.22).

This tutorial outlines how the algorithm can be used.

0. We ensure the use of :class:`~qiskit.opflow.primitive_ops.PauliSumOp` (this is the default value
   of this setting for now but we enforce it here to ensure stability of this guide as long as the
   :class:`~qiskit.algorithms.minimum_eigensolvers.AdaptVQE` class is not yet guaranteed to handle
   the :class:`~qiskit.quantum_info.SparsePauliOp` successor properly):

.. testcode::

   from qiskit_nature import settings

   settings.use_pauli_sum_op = True

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
    vqe.initial_point = np.zeros(ansatz.num_parameters)

5. We setup the :class:`~qiskit.algorithms.minimum_eigensolvers.AdaptVQE`:

.. testcode::

    from qiskit.algorithms.minimum_eigensolvers import AdaptVQE
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
