# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test MP2 Info """

import unittest
import numpy as np

from ddt import ddt, file_data
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.drivers.molecule import Molecule
from qiskit_nature.drivers.second_quantization.electronic_structure_molecule_driver import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.problems.second_quantization.electronic.electronic_structure_problem import (
    ElectronicStructureProblem,
)

from test import QiskitNatureTestCase
from qiskit_nature.settings import settings
from qiskit_nature.initializers import MP2Initializer


@ddt
class TestMP2Initializer(QiskitNatureTestCase):
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

    @file_data("./resources/test_data_mp2_initializer.json")
    def test_mp2_initializer(
        self,
        atom1,
        atom2,
        distance,
        initial_point,
        energy_delta,
        energy_deltas,
        absolute_energy,
        excitations,
    ):
        molecule = Molecule(geometry=[[atom1, [0.0, 0.0, 0.0]], [atom2, [0.0, 0.0, distance]]])

        try:
            driver = ElectronicStructureMoleculeDriver(
                molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
            )
        except QiskitNatureError:
            self.skipTest("PYSCF driver does not appear to be installed")

        problem = ElectronicStructureProblem(driver)

        # generate the second-quantized operators
        problem.second_q_ops()

        driver_result = problem.grouped_property_transformed

        particle_number = driver_result.get_property("ParticleNumber")
        electronic_energy = driver_result.get_property("ElectronicEnergy")

        num_spin_orbitals = particle_number.num_spin_orbitals

        # In practice need to build ansatz to generate excitations
        # for unit tests, load these from file
        mp2_init = MP2Initializer(
            num_spin_orbitals,
            electronic_energy,
            excitations,
        )

        with self.subTest("test num spin orbitals"):
            np.testing.assert_array_almost_equal(
                mp2_init.num_spin_orbitals, num_spin_orbitals, decimal=6
            )

        with self.subTest("Test MP2 coefficients"):
            np.testing.assert_array_almost_equal(
                mp2_init.initial_point, initial_point, decimal=6
            )

        with self.subTest("Test MP2 energy corrections"):
            np.testing.assert_array_almost_equal(
                mp2_init.energy_deltas, energy_deltas, decimal=6
            )

        with self.subTest("test overall energy correction"):
            np.testing.assert_array_almost_equal(
                mp2_init.energy_delta, energy_delta, decimal=6
            )

        with self.subTest("test absolute energy"):
            np.testing.assert_array_almost_equal(
                mp2_init.absolute_energy, absolute_energy, decimal=6
            )


if __name__ == "__main__":
    unittest.main()
