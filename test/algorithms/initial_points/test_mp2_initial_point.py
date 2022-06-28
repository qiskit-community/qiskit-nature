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
from ddt import ddt, file_data

from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_nature.drivers.molecule import Molecule
from qiskit_nature.drivers.second_quantization.electronic_structure_molecule_driver import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.second_quantization.problems.electronic.electronic_structure_problem import (
    ElectronicStructureProblem,
)
from qiskit_nature.properties.second_quantization.electronic.integrals.electronic_integrals import (
    ElectronicIntegrals,
)
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)
from qiskit_nature.settings import settings
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.circuit.library import UCC
from qiskit_nature.algorithms.initial_points import MP2InitialPoint


@ddt
class TestMP2InitialPoint(QiskitNatureTestCase):
    """Test MP2InitialPoint."""

    def setUp(self):
        super().setUp()
        settings.dict_aux_operators = True

        self.excitation_list = [[[0], [1]]]
        self.mock_ansatz = Mock(spec=UCC)
        self.mock_ansatz.excitation_list = self.excitation_list

        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.orbital_energies = Mock(spec=np.ndarray)
        electronic_energy.reference_energy = 123.45
        electronic_integrals = Mock(spec=ElectronicIntegrals)
        electronic_integrals.get_matrix = Mock(return_value=[0])
        electronic_energy.get_electronic_integral = Mock(return_value=electronic_integrals)
        self.mock_grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        self.mock_grouped_property.get_property = Mock(return_value=electronic_energy)

    def test_no_threshold(self):
        """Test when no threshold is provided."""

        mp2_initial_point = MP2InitialPoint(threshold=None)
        self.assertEqual(mp2_initial_point.threshold, 0.0)

    def test_negative_threshold(self):
        """Test when a negative threshold is provided."""

        mp2_initial_point = MP2InitialPoint(threshold=-3.0)
        self.assertEqual(mp2_initial_point.threshold, 3.0)

    def test_no_grouped_property_and_no_ansatz(self):
        """Test when no grouped property and no ansatz are provided."""

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=None, grouped_property=None)

    def test_no_grouped_property(self):
        """Test when no grouped property is provided."""

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=self.mock_ansatz, grouped_property=None)

    def test_no_ansatz(self):
        """Test when no ansatz is provided."""

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=None, grouped_property=self.mock_grouped_property)

    def test_no_electronic_energy(self):
        """Test when the electronic energy is missing."""

        self.mock_grouped_property.get_property = Mock(return_value=None)
        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(
                ansatz=self.mock_ansatz, grouped_property=self.mock_grouped_property
            )

    def test_no_two_body_mo_integrals(self):
        """Test when the two body MO integrals are missing."""

        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.orbital_energies = Mock(np.ndarray)
        electronic_energy.get_electronic_integral = Mock(return_value=None)
        self.mock_grouped_property.get_property = Mock(return_value=electronic_energy)

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(
                ansatz=self.mock_ansatz, grouped_property=self.mock_grouped_property
            )

    def test_no_orbital_energies(self):
        """Test when the orbital energies are missing."""

        electronic_integrals = Mock(spec=ElectronicIntegrals)
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.get_electronic_integral = Mock(return_value=electronic_integrals)
        electronic_energy.orbital_energies = None
        self.mock_grouped_property.get_property = Mock(return_value=electronic_energy)

        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = self.excitation_list

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(ansatz=ansatz, grouped_property=self.mock_grouped_property)

    def test_set_excitations_directly(self):
        """Test when setting excitations directly."""

        mp2_initial_point = MP2InitialPoint()
        mp2_initial_point.excitation_list = self.excitation_list
        mp2_initial_point.compute(ansatz=None, grouped_property=self.mock_grouped_property)
        self.assertEqual(mp2_initial_point.excitation_list, self.excitation_list)
        self.assertEqual(mp2_initial_point.to_numpy_array(), [0.0])

    def test_compute(self):
        """Test when grouped_property and ansatz are set via compute."""

        mp2_initial_point = MP2InitialPoint()
        mp2_initial_point.compute(
            ansatz=self.mock_ansatz, grouped_property=self.mock_grouped_property
        )

        with self.subTest("Test grouped property is set."):
            self.assertEqual(self.mock_grouped_property, mp2_initial_point.grouped_property)
        with self.subTest("Test ansatz is set."):
            self.assertEqual(self.mock_ansatz, mp2_initial_point.ansatz)
        with self.subTest("Test initial_point array is computed."):
            np.testing.assert_array_equal(mp2_initial_point.to_numpy_array(), [0.0])
        with self.subTest("Test initial_point array is computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.to_numpy_array(), [0.0])
        with self.subTest("Test energy corrections are computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.get_energy_corrections(), [0.0])
        with self.subTest("Test energy correction is computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.get_energy_correction(), 0.0)
        with self.subTest("Test energy is computed on demand."):
            mp2_initial_point._corrections = None
            np.testing.assert_array_equal(mp2_initial_point.get_energy(), 123.45)

    def test_raises_error_for_non_restricted_spins(self):
        """Test when grouped_property and ansatz are set via compute."""

        def get_matrix(value: int | None = None) -> np.ndarray:
            matrix = [1] if value == 2 else [0]
            return np.asarray(matrix)

        electronic_integrals = Mock(spec=ElectronicIntegrals)
        electronic_integrals.get_matrix = get_matrix
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.orbital_energies = Mock(spec=np.ndarray)
        electronic_energy.get_electronic_integral = Mock(return_value=electronic_integrals)
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=electronic_energy)

        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = self.excitation_list

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(NotImplementedError):
            mp2_initial_point.compute(ansatz=ansatz, grouped_property=grouped_property)

    @file_data("./resources/test_mp2_initial_point.json")
    def test_mp2_initial_point_with_real_molecules(
        self,
        atom1,
        atom2,
        distance,
        coefficients,
        energy_correction,
        energy_corrections,
        energy,
        excitations,
    ):
        """Test MP2InitialPoint with real molecules.

        Full excitation sequences generated using:

        .. code-block:: python

            converter = QubitConverter(JordanWignerMapper()
            ansatz = UCCSD(
                qubit_converter=converter,
                num_particles=num_particles,
                num_spin_orbitals=num_spin_orbitals,
            )
            _ = ansatz.operators
            excitations = ansatz.excitation_list
        """

        molecule = Molecule(geometry=[[atom1, [0.0, 0.0, 0.0]], [atom2, [0.0, 0.0, distance]]])

        try:
            driver = ElectronicStructureMoleculeDriver(
                molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
            )
            problem = ElectronicStructureProblem(driver)
            problem.second_q_ops()
        except MissingOptionalLibraryError:
            self.skipTest("PySCF driver does not appear to be installed.")

        driver_result = problem.grouped_property_transformed

        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = excitations

        mp2_initial_point = MP2InitialPoint()
        mp2_initial_point.grouped_property = driver_result
        mp2_initial_point.ansatz = ansatz

        with self.subTest("Test MP2 initial point array."):
            np.testing.assert_array_almost_equal(
                mp2_initial_point.to_numpy_array(), coefficients, decimal=6
            )

        with self.subTest("Test MP2 energy corrections."):
            np.testing.assert_array_almost_equal(
                mp2_initial_point.get_energy_corrections(), energy_corrections, decimal=6
            )

        with self.subTest("Test overall MP2 energy correction."):
            np.testing.assert_array_almost_equal(
                mp2_initial_point.get_energy_correction(), energy_correction, decimal=6
            )

        with self.subTest("Test absolute MP2 energy."):
            np.testing.assert_array_almost_equal(mp2_initial_point.get_energy(), energy, decimal=6)


if __name__ == "__main__":
    unittest.main()
