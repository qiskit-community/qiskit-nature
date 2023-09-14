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

""" Test GroundStateEigensolver """

import contextlib
import copy
import io
import unittest

from test import QiskitNatureTestCase

import numpy as np

from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator
from qiskit.test import slow_test

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers import TaperedQubitMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
class TestGroundStateEigensolverMapper(QiskitNatureTestCase):
    """Test GroundStateEigensolver with Mapper"""

    def setUp(self):
        super().setUp()
        self.driver = PySCFDriver()
        self.seed = 56
        algorithm_globals.random_seed = self.seed

        self.reference_energy = -1.1373060356951838

        self.mapper = JordanWignerMapper()
        self.tapered_mapper = TaperedQubitMapper(self.mapper)
        self.electronic_structure_problem = self.driver.run()

        self.num_spatial_orbitals = 2
        self.num_particles = (1, 1)
        self.mp2_initial_point = [0.0, 0.0, -0.07197145]

    def test_npme(self):
        """Test NumPyMinimumEigensolver"""
        solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_npme_with_default_filter(self):
        """Test NumPyMinimumEigensolver with default filter"""
        solver = NumPyMinimumEigensolver()
        solver.filter_criterion = self.electronic_structure_problem.get_default_filter_criterion()
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd(self):
        """Test VQE UCCSD case"""
        ansatz = UCC(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            "d",
            self.mapper,
            initial_state=HartreeFock(
                self.electronic_structure_problem.num_spatial_orbitals,
                self.electronic_structure_problem.num_particles,
                self.mapper,
            ),
        )
        solver = VQE(Estimator(), ansatz, SLSQP())
        solver.initial_point = [0] * ansatz.num_parameters
        calc = GroundStateEigensolver(self.mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd_taper(self):
        """Test VQE UCCSD case with TaperedQubitMapper"""
        ansatz = UCC(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            "d",
            self.mapper,
            initial_state=HartreeFock(
                self.electronic_structure_problem.num_spatial_orbitals,
                self.electronic_structure_problem.num_particles,
                self.mapper,
            ),
        )
        solver = VQE(Estimator(), ansatz, SLSQP())
        solver.initial_point = [0] * ansatz.num_parameters
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd_with_callback(self):
        """Test VQE UCCSD with callback."""

        def callback(nfev, parameters, energy, stddev):
            # pylint: disable=unused-argument
            print(f"iterations {nfev}: energy: {energy}")

        ansatz = UCC(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            "d",
            self.mapper,
            initial_state=HartreeFock(
                self.electronic_structure_problem.num_spatial_orbitals,
                self.electronic_structure_problem.num_particles,
                self.mapper,
            ),
        )
        solver = VQE(Estimator(), ansatz, SLSQP(), callback=callback)
        solver.initial_point = [0] * ansatz.num_parameters
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)
        for idx, line in enumerate(out.getvalue().split("\n")):
            if line.strip():
                self.assertTrue(line.startswith(f"iterations {idx+1}: energy: "))

    def test_aux_ops_reusability(self):
        """Test that the auxiliary operators can be reused"""
        # Regression test against #1475
        solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.tapered_mapper, solver)

        modes = 4
        h_1 = np.eye(modes, dtype=complex)
        h_2 = np.zeros((modes, modes, modes, modes))
        aux_ops = [ElectronicEnergy.from_raw_integrals(h_1, h_2).second_q_op()]
        aux_ops_copy = copy.deepcopy(aux_ops)

        _ = calc.solve(self.electronic_structure_problem)

        assert all(
            frozenset(a.items()) == frozenset(b.items()) for a, b in zip(aux_ops, aux_ops_copy)
        )

    def _setup_evaluation_operators(self):
        # first we run a ground state calculation
        ansatz = UCCSD(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                self.electronic_structure_problem.num_spatial_orbitals,
                self.electronic_structure_problem.num_particles,
                self.mapper,
            ),
        )
        solver = VQE(Estimator(), ansatz, SLSQP())
        solver.initial_point = [0] * ansatz.num_parameters
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)

        # now we decide that we want to evaluate another operator
        # for testing simplicity, we just use some pre-constructed auxiliary operators
        _, second_q_ops = self.electronic_structure_problem.second_q_ops()
        aux_ops_dict = self.tapered_mapper.map(second_q_ops)
        return calc, res, aux_ops_dict

    def _prepare_uccsd_hf(self, tapered_mapper):
        initial_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, tapered_mapper)
        ansatz = UCCSD(
            self.num_spatial_orbitals,
            self.num_particles,
            tapered_mapper,
            initial_state=initial_state,
        )

        return ansatz

    def test_uccsd_hf(self):
        """uccsd hf test"""
        ansatz = self._prepare_uccsd_hf(self.tapered_mapper)

        optimizer = SLSQP(maxiter=100)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(self.tapered_mapper, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)

    @slow_test
    def test_uccsd_hf_qasm(self):
        """uccsd hf test with qasm simulator."""
        tapered_mapper = TaperedQubitMapper(ParityMapper())
        ansatz = self._prepare_uccsd_hf(tapered_mapper)

        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(tapered_mapper, solver)

        result = gsc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(result.total_energies[0], -1.138, places=2)

    def test_freeze_core_z2_symmetry_compatibility(self):
        """Regression test against #192.

        An issue arose when the FreezeCoreTransformer was combined with the automatic Z2Symmetry
        reduction. This regression test ensures that this behavior remains fixed.
        """
        driver = PySCFDriver(
            atom="LI 0 0 0; H 0 0 1.6",
        )
        problem = FreezeCoreTransformer().transform(driver.run())
        num_particles = problem.num_particles
        tapered_mapper = problem.get_tapered_mapper(ParityMapper(num_particles))

        solver = NumPyMinimumEigensolver()
        gsc = GroundStateEigensolver(tapered_mapper, solver)

        result = gsc.solve(problem)
        self.assertAlmostEqual(result.total_energies[0], -7.882, places=2)

    def test_total_dipole(self):
        """Regression test against #198.

        An issue with calculating the dipole moment that had division None/float.
        """
        solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_dipole_moment_in_debye[0], 0.0, places=1)

    def test_print_result(self):
        """Regression test against #198 and general issues with printing results."""
        solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        res.formatting_precision = 6
        with contextlib.redirect_stdout(io.StringIO()) as out:
            print(res)
        # do NOT change the below! Lines have been truncated as to not force exact numerical matches
        expected = """\
            === GROUND STATE ENERGY ===

            * Electronic ground state energy (Hartree): -1.857275
              - computed part:      -1.857275
            ~ Nuclear repulsion energy (Hartree): 0.719969
            > Total ground state energy (Hartree): -1.137306

            === MEASURED OBSERVABLES ===

              0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000

            === DIPOLE MOMENTS ===

            ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.388949]

              0:
              * Electronic dipole moment (a.u.): [0.0  0.0  1.388949]
                - computed part:      [0.0  0.0  1.388949]
              > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                             (debye): [0.0  0.0  0.0]  Total: 0.0
        """
        for truth, expected in zip(out.getvalue().split("\n"), expected.split("\n")):
            assert truth.strip().startswith(expected.strip())

    def test_vqe_ucc_with_mp2(self):
        """Test when using MP2InitialPoint to generate the initial point."""
        ansatz = UCCSD(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                self.electronic_structure_problem.num_spatial_orbitals,
                self.electronic_structure_problem.num_particles,
                self.mapper,
            ),
        )
        solver = VQE(Estimator(), ansatz, SLSQP())

        initial_point = MP2InitialPoint()
        initial_point.ansatz = ansatz
        initial_point.problem = self.electronic_structure_problem

        solver.initial_point = initial_point.to_numpy_array()

        calc = GroundStateEigensolver(self.tapered_mapper, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)


if __name__ == "__main__":
    unittest.main()
