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
from qiskit_nature.algorithms import MP2PointGenerator


@ddt
class TestMP2PointGenerator(QiskitNatureTestCase):
    """Test MP2 initializer class.

    Full excitation sequences generated using:

    converter = QubitConverter(JordanWignerMapper()
    ansatz = UCCSD(
        qubit_converter=converter,
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
    )
    ansatz._build()
    excitations = ansatz.excitation_list
    """

    def setUp(self):
        super().setUp()
        settings.dict_aux_operators = True

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
        """Test MP2 PointGenerator with several real molecules."""

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

        particle_number = driver_result.get_property("ParticleNumber")
        electronic_energy = driver_result.get_property("ElectronicEnergy")

        num_spin_orbitals = particle_number.num_spin_orbitals
        num_orbitals = num_spin_orbitals // 2

        # In practice need to build ansatz to generate excitations
        # for unit tests, load these from file
        mp2 = MP2PointGenerator(
            num_spin_orbitals,
            electronic_energy,
            excitations,
        )

        if atom1 == "H" and atom2 == "H":
            # For molecule-independent tests, just test once for h2.

            with self.subTest("Test missing orbital energies raises error") and self.assertRaises(
                ValueError
            ):
                electronic_energy_missing = driver_result.get_property("ElectronicEnergy")
                electronic_energy_missing.orbital_energies = None
                mp2 = MP2PointGenerator(
                    num_spin_orbitals,
                    electronic_energy_missing,
                    excitations,
                )

        with self.subTest("Test number of molecular orbitals"):
            np.testing.assert_array_almost_equal(mp2.num_orbitals, num_orbitals, decimal=6)

        with self.subTest("Test number of spin orbitals"):
            np.testing.assert_array_almost_equal(
                mp2.num_spin_orbitals, num_spin_orbitals, decimal=6
            )

        with self.subTest("Test MP2 initial points"):
            np.testing.assert_array_almost_equal(mp2.initial_point, initial_point, decimal=6)

        with self.subTest("Test MP2 energy deltas"):
            np.testing.assert_array_almost_equal(mp2.energy_deltas, energy_deltas, decimal=6)

        with self.subTest("Test overall energy delta"):
            np.testing.assert_array_almost_equal(mp2.energy_delta, energy_delta, decimal=6)

        with self.subTest("Test absolute energy"):
            np.testing.assert_array_almost_equal(mp2.energy, energy, decimal=6)

        with self.subTest("Test absolute energy"):
            np.testing.assert_array_almost_equal(mp2.energy, energy, decimal=6)


if __name__ == "__main__":
    unittest.main()
