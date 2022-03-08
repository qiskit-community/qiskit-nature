# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test MP2 Info """

import ast
import unittest
import numpy as np
from ddt import data, ddt, unpack

from test import QiskitNatureTestCase
from qiskit_nature.settings import settings
from qiskit_nature import QiskitNatureError
from qiskit_nature.initializers import MP2Initializer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit.algorithms import VQE
from qiskit import Aer
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit_nature.circuit.library import UCCSD
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem


@ddt
class TestMP2Initializer(QiskitNatureTestCase):
    """Test Mp2 Info class - uses PYSCF drive to get molecule."""

    def setUp(self):
        super().setUp()

        settings.dict_aux_operators = True

        try:
            # molecule = Molecule(
            #     geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.7]]], charge=0, multiplicity=1
            # )

            molecule = Molecule(
                geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.6]]], charge=0, multiplicity=1
            )

            driver = ElectronicStructureMoleculeDriver(
                molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
            )
        except QiskitNatureError:
            self.skipTest("PySCF driver does not appear to be installed.")

        problem = ElectronicStructureProblem(driver)

        # Generate the second-quantized operators.
        _ = problem.second_q_ops()

        driver_result = problem.grouped_property_transformed

        particle_number = driver_result.get_property("ParticleNumber")
        electronic_energy = driver_result.get_property("ElectronicEnergy")

        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals

        converter = QubitConverter(JordanWignerMapper())

        self.mp2_initializer = MP2Initializer(electronic_energy)

        ansatz = UCCSD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            initializer=self.mp2_initializer,
        )

        ansatz._build()

    def test_mp2_delta(self):
        """mp2 delta test"""
        self.assertAlmostEqual(
            -0.012903900586859602, self.mp2_initializer.energy_correction, places=6
        )

    def test_mp2_energy(self):
        """mp2 energy test"""
        self.assertAlmostEqual(-7.874768670395503, self.mp2_initializer.absolute_energy, places=6)

    def test_num_coeffs_match(self):
        """test"""
        coefficients = self.mp2_initializer.coefficients
        excitations = self.mp2_initializer.excitations
        self.assertEqual(len(coefficients), len(excitations))

    def test_num_doubles(self):
        coeffs = self.mp2_initializer.coefficients
        excitations = self.mp2_initializer.excitations
        doubles = [
            coeff for coeff, excitation in zip(coeffs, excitations) if len(excitation[0]) == 2
        ]

        self.assertEqual(76, len(doubles))

    # def test_terms_frozen_core(self):
    #     """ mp2 terms frozen core test """
    #     terms = self.mp2_initializer.terms(True)
    #     self.assertEqual(16, len(terms.keys()))

    # def test_terms_frozen_core_orbital_reduction(self):
    #     """ mp2 terms frozen core orbital reduction test """
    #     terms = self.mp2_initializer.terms(True, [-3, -2])
    #     self.assertEqual(4, len(terms.keys()))

    # def test_mp2_get_term_info(self):
    #     """ mp2 get term info test """
    #     excitations = [[0, 1, 5, 9], [0, 4, 5, 9]]
    #     coeffs, e_deltas = self.mp2_initializer.mp2_get_term_info(excitations, True)
    #     np.testing.assert_array_almost_equal([0.028919010908783453, -0.07438748755263687],
    #                                          coeffs, decimal=6)
    #     np.testing.assert_array_almost_equal([-0.0010006159224579285, -0.009218577508137853],
    #                                          e_deltas, decimal=6)

    # def test_mp2_h2(self):
    #     """ Just one double excitation expected - see issue 1151 """
    #     driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.7", unit=UnitsType.ANGSTROM,
    #                          charge=0, spin=0, basis='sto3g')
    #     molecule = driver.run()

    #     mp2_initializer = MP2Info(molecule)
    #     terms = mp2_initializer.terms()
    #     self.assertEqual(1, len(terms.keys()))
    #     np.testing.assert_array_almost_equal([-0.06834019757197064, -0.012232934733533095],
    #                                          terms['0_1_2_3'], decimal=6)


if __name__ == "__main__":
    unittest.main()
