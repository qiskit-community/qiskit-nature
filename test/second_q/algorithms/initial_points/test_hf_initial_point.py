# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test HFInitialPoint."""

import unittest
from unittest.mock import Mock

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy


class TestHFInitialPoint(QiskitNatureTestCase):
    """Test HFInitialPoint."""

    def setUp(self) -> None:
        super().setUp()
        self.hf_initial_point = HFInitialPoint()
        self.ansatz = Mock(spec=UCC)
        self.ansatz.reps = 1
        self.excitation_list = [((0,), (1,))]
        self.ansatz.excitation_list = self.excitation_list

    def test_missing_ansatz(self):
        """Test set get ansatz."""
        with self.assertRaises(QiskitNatureError):
            self.hf_initial_point.compute()

    def test_set_get_ansatz(self):
        """Test set get ansatz."""
        self.hf_initial_point.ansatz = self.ansatz
        self.assertEqual(self.hf_initial_point.ansatz, self.ansatz)

    def test_set_get_problem(self):
        """Test set get problem."""
        reference_energy = 123.0
        electronic_energy = Mock(spec=ElectronicEnergy)
        problem = Mock(spec=ElectronicStructureProblem)
        problem.hamiltonian = electronic_energy
        problem.reference_energy = reference_energy
        self.hf_initial_point.problem = problem
        self.assertEqual(self.hf_initial_point.problem, problem)
        self.assertEqual(self.hf_initial_point._reference_energy, reference_energy)

    def test_set_missing_electronic_energy(self):
        """Test set missing ElectronicEnergy."""
        problem = Mock(spec=ElectronicStructureProblem)
        problem.hamiltonian = None
        with self.assertWarns(UserWarning):
            self.hf_initial_point.problem = problem
        self.assertEqual(self.hf_initial_point.problem, None)

    def test_compute(self):
        """Test length of HF initial point array."""
        problem = Mock(spec=ElectronicStructureProblem)
        problem.hamiltonian = Mock(spec=ElectronicEnergy)
        problem.reference_energy = None
        self.hf_initial_point.compute(ansatz=self.ansatz, problem=problem)
        initial_point = self.hf_initial_point.to_numpy_array()
        np.testing.assert_equal(initial_point, np.asarray([0.0]))

    def test_hf_initial_point_is_all_zero(self):
        """Test HF initial point is all zero."""
        self.hf_initial_point.ansatz = self.ansatz
        initial_point = self.hf_initial_point.to_numpy_array()
        np.testing.assert_array_equal(initial_point, np.asarray([0.0]))

    def test_hf_energy(self):
        """Test HF energy."""
        reference_energy = 123.0
        electronic_energy = Mock(spec=ElectronicEnergy)
        problem = Mock(spec=ElectronicStructureProblem)
        problem.hamiltonian = electronic_energy
        problem.reference_energy = reference_energy
        self.hf_initial_point.problem = problem
        self.hf_initial_point.ansatz = self.ansatz
        energy = self.hf_initial_point.total_energy
        self.assertEqual(energy, 123.0)


if __name__ == "__main__":
    unittest.main()
