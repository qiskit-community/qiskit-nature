# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
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

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator
from qiskit.test import slow_test

from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, SUCCD, PUCCD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
import qiskit_nature.optionals as _optionals

# pylint: disable=invalid-name


class TestUCCSDHartreeFock(QiskitNatureTestCase):
    """Test for these extensions."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 42
        self.driver = PySCFDriver(atom="H 0 0 0.735; H 0 0 0", basis="631g")

        self.electronic_structure_problem = FreezeCoreTransformer().transform(self.driver.run())

        self.mapper = self.electronic_structure_problem.get_tapered_mapper(
            ParityMapper(num_particles=self.electronic_structure_problem.num_particles)
        )

        self.num_spatial_orbitals = 4
        self.num_particles = (1, 1)

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
        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, self.mapper)

        ansatz = PUCCD(
            self.num_spatial_orbitals,
            self.num_particles,
            self.mapper,
            initial_state=initial_state,
        )

        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(self.mapper, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy_pUCCD, places=6)

    @slow_test
    def test_uccsd_hf_qUCCD0(self):
        """singlet uccd test"""
        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, self.mapper)

        ansatz = SUCCD(
            self.num_spatial_orbitals,
            self.num_particles,
            self.mapper,
            initial_state=initial_state,
        )

        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(self.mapper, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy_UCCD0, places=6)

    @slow_test
    def test_uccsd_hf_qUCCD0full(self):
        """singlet full uccd test"""
        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, self.mapper)

        ansatz = SUCCD(
            self.num_spatial_orbitals,
            self.num_particles,
            self.mapper,
            initial_state=initial_state,
            mirror=True,
        )

        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(self.mapper, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy_UCCD0full, places=6)


if __name__ == "__main__":
    unittest.main()
