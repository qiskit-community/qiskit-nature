How to use the ``AdaptVQE``
===========================

As of Qiskit Nature v0.5, the ``AdaptVQE`` algorithm has been migrated to Qiskit
Terra (released in v0.22).

This tutorial outlines how the algorithm can be used.

1. We obtain an ``ElectronicStructureProblem`` which we want to solve:

>>> from qiskit_nature.second_q.drivers import PySCFDriver
>>> driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
>>> problem = driver.run()

2. We setup our ``QubitConverter``:

>>> from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
>>> converter = QubitConverter(JordanWignerMapper())

3. We setup our ansatz:

>>> from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
>>> ansatz = UCCSD()
>>> ansatz.num_particles = problem.num_particles
>>> ansatz.num_spatial_orbitals = problem.num_spatial_orbitals
>>> ansatz.qubit_converter = converter
>>> initial_state = HartreeFock()
>>> initial_state.num_particles = problem.num_particles
>>> initial_state.num_spatial_orbitals = problem.num_spatial_orbitals
>>> initial_state.qubit_converter = converter
>>> ansatz.initial_state = initial_state

4. We setup a ``VQE``:

>>> import numpy as np
>>> from qiskit.algorithms.optimizers import SLSQP
>>> from qiskit.algorithms.minimum_eigensolvers import VQE
>>> from qiskit.primitives import Estimator
>>> vqe = VQE(Estimator(), ansatz, SLSQP())
>>> vqe.initial_point = np.zeros(ansatz.num_parameters)

5. We setup the ``AdaptVQE``:

>>> from qiskit.algorithms.minimum_eigensolvers import AdaptVQE
>>> adapt_vqe = AdaptVQE(vqe)
>>> adapt_vqe.supports_aux_operators = lambda: True  # temporary fix

6. We wrap everything in a ``GroundStateEigensolver``:

>>> from qiskit_nature.second_q.algorithms import GroundStateEigensolver
>>> solver = GroundStateEigensolver(converter, adapt_vqe)

7. We solve the problem:

>>> result = solver.solve(problem)
