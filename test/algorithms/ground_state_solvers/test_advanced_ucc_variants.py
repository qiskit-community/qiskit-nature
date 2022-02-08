# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UCCSD and HartreeFock extensions """

import unittest

from test import QiskitNatureTestCase

from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.test import slow_test

from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.circuit.library import HartreeFock, SUCCD, PUCCD
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
import qiskit_nature.optionals as _optionals


# pylint: disable=invalid-name


class TestUCCSDHartreeFock(QiskitNatureTestCase):
    """Test for these extensions."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        self.driver = PySCFDriver(atom="H 0 0 0.735; H 0 0 0", basis="631g")

        self.qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)

        self.electronic_structure_problem = ElectronicStructureProblem(
            self.driver, [FreezeCoreTransformer()]
        )

        self.num_spin_orbitals = 8
        self.num_particles = (1, 1)

        # because we create the initial state and ansatzes early, we need to ensure the qubit
        # converter already ran such that convert_match works as expected
        _ = self.qubit_converter.convert(
            self.electronic_structure_problem.second_q_ops()[0], self.num_particles
        )

        self.reference_energy_pUCCD = -1.1434447924298028
        self.reference_energy_UCCD0 = -1.1476045878481704
        self.reference_energy_UCCD0full = -1.1515491334334347
        # reference energy of UCCSD/VQE with tapering everywhere
        self.reference_energy_UCCSD = -1.1516142309717594
        # reference energy of UCCSD/VQE when no tapering on excitations is used
        self.reference_energy_UCCSD_no_tap_exc = -1.1516142309717594
        # excitations for succ
        self.reference_singlet_double_excitations = [
            [0, 1, 4, 5],
            [0, 1, 4, 6],
            [0, 1, 4, 7],
            [0, 2, 4, 6],
            [0, 2, 4, 7],
            [0, 3, 4, 7],
        ]
        # groups for succ_full
        self.reference_singlet_groups = [
            [[0, 1, 4, 5]],
            [[0, 1, 4, 6], [0, 2, 4, 5]],
            [[0, 1, 4, 7], [0, 3, 4, 5]],
            [[0, 2, 4, 6]],
            [[0, 2, 4, 7], [0, 3, 4, 6]],
            [[0, 3, 4, 7]],
        ]

    @slow_test
    def test_uccsd_hf_qpUCCD(self):
        """paired uccd test"""
        self.skipTest(
            "Temporarily skip test until the changes done by "
            "https://github.com/Qiskit/qiskit-terra/pull/7551 are handled properly."
        )
        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(
            self.num_spin_orbitals, self.num_particles, self.qubit_converter
        )

        ansatz = PUCCD(
            self.qubit_converter,
            self.num_particles,
            self.num_spin_orbitals,
            initial_state=initial_state,
        )

        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=QuantumInstance(backend=BasicAer.get_backend("statevector_simulator")),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy_pUCCD, places=6)

    @slow_test
    def test_uccsd_hf_qUCCD0(self):
        """singlet uccd test"""
        self.skipTest(
            "Temporarily skip test until the changes done by "
            "https://github.com/Qiskit/qiskit-terra/pull/7551 are handled properly."
        )
        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(
            self.num_spin_orbitals, self.num_particles, self.qubit_converter
        )

        ansatz = SUCCD(
            self.qubit_converter,
            self.num_particles,
            self.num_spin_orbitals,
            initial_state=initial_state,
        )

        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=QuantumInstance(backend=BasicAer.get_backend("statevector_simulator")),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy_UCCD0, places=6)

    @unittest.skip("Skip until https://github.com/Qiskit/qiskit-nature/issues/91 is closed.")
    def test_uccsd_hf_qUCCD0full(self):
        """singlet full uccd test"""
        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(
            self.num_spin_orbitals, self.num_particles, self.qubit_converter
        )

        # TODO: add `full` option
        ansatz = SUCCD(
            self.qubit_converter,
            self.num_particles,
            self.num_spin_orbitals,
            initial_state=initial_state,
        )

        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=QuantumInstance(backend=BasicAer.get_backend("statevector_simulator")),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy_UCCD0full, places=6)


if __name__ == "__main__":
    unittest.main()
