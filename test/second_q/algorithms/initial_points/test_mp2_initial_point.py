# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test MP2InitialPoint """

from __future__ import annotations

import unittest
from unittest.mock import Mock

from test import QiskitNatureTestCase

import numpy as np
from ddt import ddt, data

from qiskit_nature import optionals
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import QubitConverter, JordanWignerMapper
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint


@ddt
class TestMP2InitialPoint(QiskitNatureTestCase):
    """Test MP2InitialPoint."""

    def setUp(self):
        super().setUp()
        self.excitation_list = [[[0], [1]]]
        self.mock_ansatz = Mock(spec=UCC)
        self.mock_ansatz.reps = 1
        self.mock_ansatz.excitation_list = self.excitation_list

        self.electronic_energy = ElectronicEnergy.from_raw_integrals(
            np.zeros((2, 2)), np.zeros((2, 2, 2, 2))
        )
        self.mock_problem = ElectronicStructureProblem(self.electronic_energy)
        self.mock_problem.reference_energy = 123.45
        self.mock_problem.orbital_energies = np.asarray([])
        self.mock_problem.num_particles = (1, 1)

    @unittest.skipIf(not optionals.HAS_PYSCF, "pyscf not available.")
    @data("H 0 0 0; H 0 0 0.7", "Li 0 0 0; H 0 0 1.6")
    def test_mp2_initial_point_with_real_molecules(
        self,
        atom,
    ):
        """Test MP2InitialPoint with real molecules."""
        from pyscf import gto  # pylint: disable=import-error

        # Compute the PySCF result
        pyscf_mol = gto.M(atom=atom, basis="sto3g", verbose=0)
        pyscf_mp = pyscf_mol.MP2().run(verbose=0)

        driver = PySCFDriver(atom=atom, basis="sto3g")

        problem = driver.run()
        problem.second_q_ops()

        num_particles = (problem.num_alpha, problem.num_beta)
        num_spatial_orbitals = problem.num_spatial_orbitals

        qubit_converter = QubitConverter(mapper=JordanWignerMapper())

        initial_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_converter=qubit_converter,
        )
        ansatz = UCC(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations="sd",
            qubit_converter=qubit_converter,
            initial_state=initial_state,
        )

        mp2_initial_point = MP2InitialPoint()
        mp2_initial_point.problem = problem
        mp2_initial_point.ansatz = ansatz

        with self.subTest("Test the MP2 energy correction."):
            np.testing.assert_almost_equal(
                mp2_initial_point.energy_correction, pyscf_mp.e_corr, decimal=4
            )

        with self.subTest("Test the total MP2 energy."):
            np.testing.assert_almost_equal(
                mp2_initial_point.total_energy, pyscf_mp.e_tot, decimal=4
            )

        with self.subTest("Test the T2 amplitudes."):
            mp2_initial_point.compute()
            np.testing.assert_array_almost_equal(
                mp2_initial_point.t2_amplitudes, pyscf_mp.t2, decimal=4
            )

    def test_no_threshold(self):
        """Test when no threshold is provided."""

        mp2_initial_point = MP2InitialPoint(threshold=None)
        self.assertEqual(mp2_initial_point.threshold, 0.0)

    def test_negative_threshold(self):
        """Test when a negative threshold is provided."""

        mp2_initial_point = MP2InitialPoint(threshold=-3.0)
        self.assertEqual(mp2_initial_point.threshold, 3.0)

    def test_no_problem_and_no_ansatz(self):
        """Test when no problem and no ansatz are provided."""

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=None, problem=None)

    def test_no_problem(self):
        """Test when no problem is provided."""

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=None)

    def test_no_ansatz(self):
        """Test when no ansatz is provided."""

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=None, problem=self.mock_problem)

    def test_no_electronic_energy(self):
        """Test when the electronic energy is missing."""
        mp2_initial_point = MP2InitialPoint()
        problem = Mock(spec=ElectronicStructureProblem)
        problem.hamiltonian = None
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=problem)

    def test_no_two_body_mo_integrals(self):
        """Test when the two body MO integrals are missing."""

        electronic_energy = ElectronicEnergy.from_raw_integrals(np.zeros((2, 2)), None)
        problem = ElectronicStructureProblem(electronic_energy)
        problem.orbital_energies = np.asarray([])

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=problem)

    def test_no_orbital_energies(self):
        """Test when the orbital energies are missing."""

        electronic_energy = ElectronicEnergy.from_raw_integrals(
            np.zeros((2, 2)), np.zeros((2, 2, 2, 2))
        )
        problem = ElectronicStructureProblem(electronic_energy)
        problem.orbital_energies = None

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=problem)

    def test_no_particle_number(self):
        """Test when the particle number is missing."""

        mp2_initial_point = MP2InitialPoint()
        self.mock_problem.num_particles = None
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=self.mock_problem)

    def test_compute(self):
        """Test when problem and ansatz are set via compute."""

        mp2_initial_point = MP2InitialPoint()
        mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=self.mock_problem)

        with self.subTest("Test grouped property is set."):
            self.assertEqual(self.mock_problem, mp2_initial_point.problem)
        with self.subTest("Test ansatz is set."):
            self.assertEqual(self.mock_ansatz, mp2_initial_point.ansatz)
        with self.subTest("Test initial_point array is computed."):
            np.testing.assert_array_equal(mp2_initial_point.to_numpy_array(), [0.0])
        with self.subTest("Test initial_point array is computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.to_numpy_array(), [0.0])
        with self.subTest("Test energy correction is computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.energy_correction, 0.0)
        with self.subTest("Test energy is computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.total_energy, 123.45)

    def test_raises_error_for_non_restricted_spins(self):
        """Test when problem and ansatz are set via compute."""

        electronic_energy = ElectronicEnergy.from_raw_integrals(
            np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), np.zeros((2, 2)), np.zeros((2, 2, 2, 2))
        )
        problem = ElectronicStructureProblem(electronic_energy)
        problem.orbital_energies = Mock(spec=np.ndarray)

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(NotImplementedError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, problem=problem)


if __name__ == "__main__":
    unittest.main()
