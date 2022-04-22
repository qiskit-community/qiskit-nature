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

"""Test the MP2 Initializer for generating an initial point for VQE."""

import unittest
from unittest.mock import Mock
from qiskit_nature.properties.second_quantization.electronic.integrals.electronic_integrals import (
    ElectronicIntegrals,
)

from test import QiskitNatureTestCase

import numpy as np
from ddt import ddt, file_data

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.drivers.molecule import Molecule
from qiskit_nature.drivers.second_quantization.electronic_structure_molecule_driver import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization.electronic.electronic_structure_problem import (
    ElectronicStructureProblem,
)
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)
from qiskit_nature.settings import settings
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.circuit.library import UCC
from qiskit_nature.algorithms import MP2InitialPoint


@ddt
class TestMP2InitialPoint(QiskitNatureTestCase):
    """Test MP2 initial point.

    Full excitation sequences generated using:

    converter = QubitConverter(JordanWignerMapper()
    ansatz = UCCSD(
        qubit_converter=converter,
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
    )
    _ = ansatz.operators
    excitations = ansatz.excitation_list
    """

    def setUp(self):
        super().setUp()
        settings.dict_aux_operators = True

    def test_no_threshold(self):
        mp2_initial_point = MP2InitialPoint(threshold=None)
        self.assertEqual(mp2_initial_point.threshold, 0.0)

    def test_negative_threshold(self):
        mp2_initial_point = MP2InitialPoint(threshold=-3.0)
        self.assertEqual(mp2_initial_point.threshold, 3.0)

    def test_no_grouped_property_and_no_ansatz(self):
        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(None, None)

    def test_no_grouped_property(self):
        ansatz = Mock(spec=UCC)
        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(None, ansatz)

    def test_no_ansatz(self):
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(grouped_property, None)

    def test_not_ucc_ansatz(self):
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        mp2_initial_point = MP2InitialPoint()
        ansatz = Mock(spec=EvolvedOperatorAnsatz)
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(grouped_property, None)

    def test_no_electronic_energy(self):
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=None)
        ansatz = Mock(spec=UCC)
        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(grouped_property, ansatz)

    def test_no_two_body_mo_integrals(self):
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.orbital_energies = Mock(np.ndarray)
        electronic_energy.get_electronic_integral = Mock(return_value=None)
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=electronic_energy)

        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = [[[0], [1]]]

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(grouped_property, ansatz)

    def test_no_orbital_energies(self):
        electronic_integrals = Mock(spec=ElectronicIntegrals)
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.get_electronic_integral = Mock(return_value=electronic_integrals)
        electronic_energy.orbital_energies = None
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=electronic_energy)

        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = [[[0], [1]]]

        mp2_initial_point = MP2InitialPoint()
        with self.assertRaises(QiskitNatureError):
            mp2_initial_point.compute(grouped_property, ansatz)

    def test_set_excitations_directly(self):
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.orbital_energies = Mock(spec=np.ndarray)
        electronic_integrals = Mock(spec=ElectronicIntegrals)
        electronic_integrals.get_matrix = Mock(return_value=[0])
        electronic_energy.get_electronic_integral = Mock(return_value=electronic_integrals)
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=electronic_energy)

        mp2_initial_point = MP2InitialPoint()
        mp2_initial_point.excitation_list = [[[0], [1]]]
        mp2_initial_point.compute(grouped_property, None)
        self.assertEqual(mp2_initial_point.get_energy_correction(), 0.0)

    @file_data("./resources/test_data_mp2_point_generator.json")
    def test_mp2_point_generator(
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
        """Test MP2InitialPoint with several real molecules."""

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

        with self.subTest("MP2 initial point array"):
            np.testing.assert_array_almost_equal(
                mp2_initial_point.to_numpy_array(), coefficients, decimal=6
            )

        with self.subTest("MP2 energy corrections"):
            np.testing.assert_array_almost_equal(
                mp2_initial_point.get_energy_corrections(), energy_corrections, decimal=6
            )

        with self.subTest("overall MP2 energy correction"):
            np.testing.assert_array_almost_equal(
                mp2_initial_point.get_energy_correction(), energy_correction, decimal=6
            )

        with self.subTest("absolute MP2 energy"):
            np.testing.assert_array_almost_equal(mp2_initial_point.get_energy(), energy, decimal=6)


if __name__ == "__main__":
    unittest.main()
