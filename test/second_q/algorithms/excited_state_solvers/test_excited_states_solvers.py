# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Numerical qEOM excited states calculation """

from __future__ import annotations

import unittest

from test import QiskitNatureTestCase
from ddt import ddt, named_data
import numpy as np

from qiskit_algorithms import NumPyEigensolver, NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    QubitMapper,
    TaperedQubitMapper,
)

from qiskit_nature.second_q.algorithms import GroundStateEigensolver, ExcitedStatesEigensolver, QEOM
import qiskit_nature.optionals as _optionals


@ddt
class TestNumericalQEOMESCCalculation(QiskitNatureTestCase):
    """Test Numerical qEOM excited states calculation"""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.75",
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )

        self.reference_energies = [
            -1.8427016,
            -1.8427016 + 0.5943372,
            -1.8427016 + 0.95788352,
            -1.8427016 + 1.5969296,
        ]
        self.mapper = JordanWignerMapper()
        self.electronic_structure_problem = self.driver.run()
        self.num_particles = self.electronic_structure_problem.num_particles

        solver = NumPyEigensolver(k=10)
        self.ref = solver

    def _assert_energies(self, computed, references, *, places=4):
        with self.subTest("same number of energies"):
            self.assertEqual(len(computed), len(references))

        with self.subTest("ground state"):
            self.assertAlmostEqual(computed[0], references[0], places=places)

        for i in range(1, len(computed)):
            with self.subTest(f"{i}. excited state"):
                self.assertAlmostEqual(computed[i], references[i], places=places)

    def _compute_and_assert_qeom_energies(self, mapper: QubitMapper):
        estimator = Estimator()
        ansatz = UCCSD(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            mapper,
            initial_state=HartreeFock(
                self.electronic_structure_problem.num_spatial_orbitals,
                self.electronic_structure_problem.num_particles,
                mapper,
            ),
        )
        solver = VQE(estimator, ansatz, SLSQP())
        solver.initial_point = [0] * ansatz.num_parameters
        gsc = GroundStateEigensolver(mapper, solver)
        esc = QEOM(gsc, estimator, "sd")
        results = esc.solve(self.electronic_structure_problem)
        self._assert_energies(results.computed_energies, self.reference_energies)

    def test_numpy_mes(self):
        """Test NumPyMinimumEigenSolver with QEOM"""
        solver = NumPyMinimumEigensolver()
        gsc = GroundStateEigensolver(self.mapper, solver)
        esc = QEOM(gsc, Estimator(), "sd")
        results = esc.solve(self.electronic_structure_problem)
        self._assert_energies(results.computed_energies, self.reference_energies)

    @named_data(
        ["JWM", JordanWignerMapper()],
        ["PM", ParityMapper()],
        ["PM_TQR", ParityMapper(num_particles=(1, 1))],
    )
    def test_solve_with_vqe_mes_mapper(self, mapper: QubitMapper):
        """Test QEOM with VQE + UCCSD and various QubitMapper"""
        self._compute_and_assert_qeom_energies(mapper)

    @named_data(
        ["JW", lambda n, esp: TaperedQubitMapper(JordanWignerMapper())],
        ["JW_Z2", lambda n, esp: esp.get_tapered_mapper(JordanWignerMapper())],
        ["PM", lambda n, esp: TaperedQubitMapper(ParityMapper())],
        ["PM_Z2", lambda n, esp: esp.get_tapered_mapper(ParityMapper())],
        ["PM_TQR", lambda n, esp: TaperedQubitMapper(ParityMapper(n))],
        ["PM_TQR_Z2", lambda n, esp: esp.get_tapered_mapper(ParityMapper(n))],
    )
    def test_solve_with_vqe_mes_taperedmapper(self, tapered_mapper_creator):
        """Test QEOM with VQE + UCCSD and various QubitMapper"""
        tapered_mapper = tapered_mapper_creator(
            self.num_particles, self.electronic_structure_problem
        )
        self._compute_and_assert_qeom_energies(tapered_mapper)

    def test_numpy(self):
        """Test NumPy with ExcitedStatesEigensolver"""

        # pylint: disable=unused-argument
        def filter_criterion(eigenstate, eigenvalue, aux_values):
            return np.isclose(aux_values["ParticleNumber"][0], 2.0)

        solver = NumPyEigensolver(k=10)
        solver.filter_criterion = filter_criterion
        esc = ExcitedStatesEigensolver(self.mapper, solver)
        results = esc.solve(self.electronic_structure_problem)

        # filter duplicates from list
        computed_energies = [results.computed_energies[0]]
        for comp_energy in results.computed_energies[1:]:
            if not np.isclose(comp_energy, computed_energies[-1]):
                computed_energies.append(comp_energy)

        self._assert_energies(computed_energies, self.reference_energies)

    def test_custom_filter_criterion(self):
        """Test NumPy with ExcitedStatesEigensolver + Custom filter criterion for doublet states"""

        driver = PySCFDriver(
            atom="Be .0 .0 .0; H .0 .0 0.75",
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto3g",
        )

        transformer = ActiveSpaceTransformer((1, 2), 4)
        # We define an ActiveSpaceTransformer to reduce the duration of this test example.

        esp = transformer.transform(driver.run())

        mapper = esp.get_tapered_mapper(JordanWignerMapper())

        expected_spin = 0.75  # Doublet states
        expected_num_electrons = 3  # 1 alpha electron + 2 beta electrons

        # pylint: disable=unused-argument
        def custom_filter_criterion(eigenstate, eigenvalue, aux_values):
            num_particles_aux = aux_values["ParticleNumber"][0]
            total_angular_momentum_aux = aux_values["AngularMomentum"][0]

            return np.isclose(total_angular_momentum_aux, expected_spin) and np.isclose(
                num_particles_aux, expected_num_electrons
            )

        solver = NumPyEigensolver(k=100)
        solver.filter_criterion = custom_filter_criterion
        esc = ExcitedStatesEigensolver(mapper, solver)
        results = esc.solve(esp)

        # filter duplicates from list
        computed_energies = [results.computed_energies[0]]
        for comp_energy in results.computed_energies[1:]:
            if not np.isclose(comp_energy, computed_energies[-1]):
                computed_energies.append(comp_energy)

        ref_energies = [
            -2.6362023196223254,
            -2.2971398524128923,
            -2.2020252702733165,
            -2.1044859216523752,
            -1.696132447109807,
            -1.6416831059956618,
        ]

        self._assert_energies(computed_energies, ref_energies, places=3)


if __name__ == "__main__":
    unittest.main()
