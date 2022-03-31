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
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy

from test import QiskitNatureTestCase

import numpy as np
from ddt import ddt, file_data

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_nature.drivers.molecule import Molecule
from qiskit_nature.drivers.second_quantization.electronic_structure_molecule_driver import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization.electronic.electronic_structure_problem import (
    ElectronicStructureProblem,
)

from qiskit_nature.settings import settings
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
        self.mock_ansatz = Mock()
        self.mock_ansatz.operators = None

    def test_mp2_missing_orbital_energies(
        self,
    ):
        """Test MP2InitialPoint raises errors for bad input."""

        mp2 = MP2InitialPoint()

        # Mock up driver result with missing orbital energies.
        mock_result = Mock()
        mock_electronic_energy = Mock()
        mock_electronic_energy.orbital_energies = None
        mock_result.get_property = Mock(return_value=mock_electronic_energy)

        with self.assertRaises(ValueError):
            _ = mp2.get_initial_point(mock_result, self.mock_ansatz)

    @file_data("./resources/test_data_mp2_point_generator.json")
    def test_mp2_point_generator(
        self,
        atom1,
        atom2,
        distance,
        initial_point,
        energy_delta,
        energy_deltas,
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

        # we just need the excitation list from the ansatz; so mock it up
        self.mock_ansatz.excitation_list = excitations

        mp2 = MP2InitialPoint()

        with self.subTest("Test MP2 initial point"):
            np.testing.assert_array_almost_equal(
                mp2.get_initial_point(driver_result, self.mock_ansatz),
                initial_point,
                decimal=6,
            )

        with self.subTest("Test MP2 energy deltas"):
            np.testing.assert_array_almost_equal(mp2.energy_deltas, energy_deltas, decimal=6)

        with self.subTest("Test overall energy delta"):
            np.testing.assert_array_almost_equal(mp2.energy_delta, energy_delta, decimal=6)

        with self.subTest("Test absolute energy"):
            np.testing.assert_array_almost_equal(mp2.energy, energy, decimal=6)


if __name__ == "__main__":
    unittest.main()
