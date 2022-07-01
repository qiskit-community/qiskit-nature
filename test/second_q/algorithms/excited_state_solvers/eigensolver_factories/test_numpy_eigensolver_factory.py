# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
""" Test NumPyMinimumEigensolver Factory """
import unittest
from test import QiskitNatureTestCase
import numpy as np

from qiskit.algorithms import NumPyEigensolver
from qiskit_nature.second_q.algorithms import NumPyEigensolverFactory
from qiskit_nature.second_q.drivers import UnitsType
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
import qiskit_nature.optionals as _optionals


class TestNumPyEigensolverFactory(QiskitNatureTestCase):
    """Test NumPyMinimumEigensovler Factory"""

    # NOTE: The actual usage of this class is mostly tested in combination with the ground-state
    # eigensolvers (one module above).

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()

        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.75",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )

        self.electronic_structure_problem = ElectronicStructureProblem(self.driver)

        # pylint: disable=unused-argument
        def filter_criterion(eigenstate, eigenvalue, aux_values):
            return np.isclose(aux_values[0][0], 2.0)

        self.k = 99
        self._numpy_eigensolver_factory = NumPyEigensolverFactory(
            filter_criterion=filter_criterion, k=self.k
        )

    def test_setters_getters(self):
        """Test Getter/Setter"""

        # filter_criterion
        self.assertIsNotNone(self._numpy_eigensolver_factory.filter_criterion)

        # pylint: disable=unused-argument
        def filter_criterion(eigenstate, eigenvalue, aux_values):
            return np.isclose(aux_values[0][0], 3.0)

        self._numpy_eigensolver_factory.filter_criterion = filter_criterion
        self.assertEqual(self._numpy_eigensolver_factory.filter_criterion, filter_criterion)

        # k
        self.assertEqual(self._numpy_eigensolver_factory.k, self.k)
        self._numpy_eigensolver_factory.k = 100
        self.assertEqual(self._numpy_eigensolver_factory.k, 100)

        # use_default_filter_criterion
        self.assertFalse(self._numpy_eigensolver_factory.use_default_filter_criterion)
        self._numpy_eigensolver_factory.use_default_filter_criterion = True
        self.assertTrue(self._numpy_eigensolver_factory.use_default_filter_criterion)
        # get_solver
        solver = self._numpy_eigensolver_factory.get_solver(self.electronic_structure_problem)
        self.assertIsInstance(solver, NumPyEigensolver)
        self.assertEqual(solver.k, 100)
        self.assertEqual(solver.filter_criterion, filter_criterion)


if __name__ == "__main__":
    unittest.main()
