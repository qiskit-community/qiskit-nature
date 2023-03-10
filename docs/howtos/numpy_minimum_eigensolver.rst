.. _how-to-numpy-min:

How to use the ``NumPyMinimumEigensolver``
==========================================

In order to ensure a physically meaningful ground state of a hamiltonian is found when using the
:class:`~qiskit.algorithms.minimum_eigensolvers.NumPyMinimumEigensolver` one needs to set the
:attr:`~qiskit.algorithms.minimum_eigensolvers.NumPyMinimumEigensolver.filter_criterion` attribute
of the solver.

Subclasses of :class:`~qiskit_nature.second_q.problems.BaseProblem` in Qiskit Nature provide the
:meth:`~qiskit_nature.second_q.problems.BaseProblem.get_default_filter_criterion` method which
provides a default implementation of such a filter criterion for commonly encountered cases.

Below we show how you can use this setting.

1. We obtain an ``ElectronicStructureProblem`` which we want to solve:

.. testcode::

    from qiskit_nature.second_q.drivers import PySCFDriver
    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
    problem = driver.run()

2. We setup our ``QubitMapper``:

.. testcode::

    from qiskit_nature.second_q.mappers import JordanWignerMapper
    mapper = JordanWignerMapper()

3. We setup our ``NumPyMinimumEigensolver``:

.. testcode::

    from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    algo = NumPyMinimumEigensolver()
    algo.filter_criterion = problem.get_default_filter_criterion()

4. We wrap everything in a ``GroundStateEigensolver``:

.. testcode::

    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    solver = GroundStateEigensolver(mapper, algo)

5. We solve the problem:

.. testcode::

    result = solver.solve(problem)
