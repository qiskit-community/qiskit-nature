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

import unittest
import numpy as np

from test import QiskitNatureTestCase
from qiskit_nature.settings import settings
from qiskit_nature import QiskitNatureError
from qiskit_nature.initializers import MP2Initializer
from qiskit_nature.drivers import Molecule
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.circuit.library import UCCSD
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem


class TestMP2Initializer(QiskitNatureTestCase):
    """Test Mp2 Info class - uses PYSCF drive to get molecule.

    Full excitation sequences generated using:

    converter = QubitConverter(JordanWignerMapper()
    ansatz = UCCSD(
        qubit_converter=converter,
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initializer=mp2_initializer,

    ansatz._build()
    """

    def setUp(self):
        super().setUp()

        settings.dict_aux_operators = True

        lih_molecule = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.6]]])
        self._lih_excitations = [
            ((0,), (2,)),
            ((0,), (3,)),
            ((0,), (4,)),
            ((0,), (5,)),
            ((1,), (2,)),
            ((1,), (3,)),
            ((1,), (4,)),
            ((1,), (5,)),
            ((6,), (8,)),
            ((6,), (9,)),
            ((6,), (10,)),
            ((6,), (11,)),
            ((7,), (8,)),
            ((7,), (9,)),
            ((7,), (10,)),
            ((7,), (11,)),
            ((0, 1), (2, 3)),
            ((0, 1), (2, 4)),
            ((0, 1), (2, 5)),
            ((0, 6), (2, 8)),
            ((0, 6), (2, 9)),
            ((0, 6), (2, 10)),
            ((0, 6), (2, 11)),
            ((0, 7), (2, 8)),
            ((0, 7), (2, 9)),
            ((0, 7), (2, 10)),
            ((0, 7), (2, 11)),
            ((0, 1), (3, 4)),
            ((0, 1), (3, 5)),
            ((0, 6), (3, 8)),
            ((0, 6), (3, 9)),
            ((0, 6), (3, 10)),
            ((0, 6), (3, 11)),
            ((0, 7), (3, 8)),
            ((0, 7), (3, 9)),
            ((0, 7), (3, 10)),
            ((0, 7), (3, 11)),
            ((0, 1), (4, 5)),
            ((0, 6), (4, 8)),
            ((0, 6), (4, 9)),
            ((0, 6), (4, 10)),
            ((0, 6), (4, 11)),
            ((0, 7), (4, 8)),
            ((0, 7), (4, 9)),
            ((0, 7), (4, 10)),
            ((0, 7), (4, 11)),
            ((0, 6), (5, 8)),
            ((0, 6), (5, 9)),
            ((0, 6), (5, 10)),
            ((0, 6), (5, 11)),
            ((0, 7), (5, 8)),
            ((0, 7), (5, 9)),
            ((0, 7), (5, 10)),
            ((0, 7), (5, 11)),
            ((1, 6), (2, 8)),
            ((1, 6), (2, 9)),
            ((1, 6), (2, 10)),
            ((1, 6), (2, 11)),
            ((1, 7), (2, 8)),
            ((1, 7), (2, 9)),
            ((1, 7), (2, 10)),
            ((1, 7), (2, 11)),
            ((1, 6), (3, 8)),
            ((1, 6), (3, 9)),
            ((1, 6), (3, 10)),
            ((1, 6), (3, 11)),
            ((1, 7), (3, 8)),
            ((1, 7), (3, 9)),
            ((1, 7), (3, 10)),
            ((1, 7), (3, 11)),
            ((1, 6), (4, 8)),
            ((1, 6), (4, 9)),
            ((1, 6), (4, 10)),
            ((1, 6), (4, 11)),
            ((1, 7), (4, 8)),
            ((1, 7), (4, 9)),
            ((1, 7), (4, 10)),
            ((1, 7), (4, 11)),
            ((1, 6), (5, 8)),
            ((1, 6), (5, 9)),
            ((1, 6), (5, 10)),
            ((1, 6), (5, 11)),
            ((1, 7), (5, 8)),
            ((1, 7), (5, 9)),
            ((1, 7), (5, 10)),
            ((1, 7), (5, 11)),
            ((6, 7), (8, 9)),
            ((6, 7), (8, 10)),
            ((6, 7), (8, 11)),
            ((6, 7), (9, 10)),
            ((6, 7), (9, 11)),
            ((6, 7), (10, 11)),
        ]

        h2_molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.7]]])
        self._h2_excitations = [((0,), (1,)), ((2,), (3,)), ((0, 2), (1, 3))]

        try:
            self._lih_mp2_initializer = _generate_mp2_initializer(lih_molecule)
            self._h2_mp2_initializer = _generate_mp2_initializer(h2_molecule)
        except QiskitNatureError:
            self.skipTest("PySCF driver does not appear to be installed.")

    def test_mp2_delta(self):
        """mp2 delta test"""
        self._lih_mp2_initializer.compute_corrections(self._lih_excitations)
        self.assertAlmostEqual(
            -0.012903900586859602, self._lih_mp2_initializer.energy_correction, places=6
        )

    def test_mp2_energy(self):
        """Test MP2 energy."""
        self._lih_mp2_initializer.compute_corrections(self._lih_excitations)
        self.assertAlmostEqual(
            -7.874768670395503, self._lih_mp2_initializer.absolute_energy, places=6
        )

    def test_num_coeffs_match(self):
        """Test that the number of excitations and coefficients match."""
        self._lih_mp2_initializer.compute_corrections(self._lih_excitations)
        coefficients = self._lih_mp2_initializer.coefficients
        excitations = self._lih_mp2_initializer.excitations
        self.assertEqual(len(coefficients), len(excitations))

    def test_num_doubles(self):
        """Test that the number of excitations and coefficients match."""
        self._lih_mp2_initializer.compute_corrections(self._lih_excitations)
        coeffs = self._lih_mp2_initializer.coefficients
        excitations = self._lih_mp2_initializer.excitations
        doubles = [
            coeff for coeff, excitation in zip(coeffs, excitations) if len(excitation[0]) == 2
        ]

        self.assertEqual(76, len(doubles))

    def test_mp2_correction_values(self):
        """Test correction values for a specific subset of excitations."""
        # aqua_excitations = [[0, 1, 5, 9], [0, 4, 5, 9]]
        excitations = [((1, 7), (5, 8)), ((1, 7), (5, 11))]
        self._lih_mp2_initializer.compute_corrections(excitations)
        coeffs, e_deltas = self._lih_mp2_initializer.compute_corrections(excitations)
        np.testing.assert_array_almost_equal(
            [0.028919010908783453, -0.07438748755263687], coeffs, decimal=6
        )
        np.testing.assert_array_almost_equal(
            [-0.0010006159224579285, -0.009218577508137853], e_deltas, decimal=6
        )

    def test_mp2_h2(self):
        """Just one double excitation expected - see issue 1151"""
        excitations = [((0, 2), (1, 3))]
        coeffs, e_deltas = self._h2_mp2_initializer.compute_corrections(excitations)
        self.assertEqual(1, len(coeffs))
        np.testing.assert_array_almost_equal(
            [-0.06834019757197064, -0.012232934733533095], [coeffs[0], e_deltas[0]], decimal=6
        )

    # TODO transform orbitals in the new way
    # def test_terms_frozen_core(self):
    #     """ mp2 terms frozen core test """
    #     terms = self.mp2_initializer.terms(True)
    #     self.assertEqual(16, len(terms.keys()))

    # def test_terms_frozen_core_orbital_reduction(self):
    #     """ mp2 terms frozen core orbital reduction test """
    #     terms = self.mp2_initializer.terms(True, [-3, -2])
    #     self.assertEqual(4, len(terms.keys()))


def _generate_mp2_initializer(molecule: Molecule) -> MP2Initializer:

    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
    )

    problem = ElectronicStructureProblem(driver)

    # Generate the second-quantized operators.
    _ = problem.second_q_ops()

    driver_result = problem.grouped_property_transformed

    particle_number = driver_result.get_property("ParticleNumber")
    electronic_energy = driver_result.get_property("ElectronicEnergy")

    num_particles = (particle_number.num_alpha, particle_number.num_beta)
    num_spin_orbitals = particle_number.num_spin_orbitals

    mp2_initializer = MP2Initializer(num_spin_orbitals, electronic_energy)

    return mp2_initializer


if __name__ == "__main__":
    unittest.main()
