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

""" Test GroundStateEigensolver """

import contextlib
import copy
import io
import unittest

from test import QiskitNatureTestCase

import numpy as np

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.primitives import Estimator
from qiskit.test import slow_test
from qiskit.utils import algorithm_globals

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    VQEUCCFactory,
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
class TestGroundStateEigensolver(QiskitNatureTestCase):
    """Test GroundStateEigensolver"""

    def setUp(self):
        super().setUp()
        self.driver = PySCFDriver()
        self.seed = 56
        algorithm_globals.random_seed = self.seed

        self.reference_energy = -1.1373060356951838

        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = self.driver.run()

        self.num_spatial_orbitals = 2
        self.num_particles = (1, 1)
        self.mp2_initial_point = [0.0, 0.0, -0.07197145]

    def test_npme(self):
        """Test NumPyMinimumEigensolver"""
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_npme_with_default_filter(self):
        """Test NumPyMinimumEigensolver with default filter"""
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd(self):
        """Test VQE UCCSD case"""
        solver = VQEUCCFactory(Estimator(), UCC(excitations="d"), SLSQP())
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd_with_callback(self):
        """Test VQE UCCSD with callback."""

        def callback(nfev, parameters, energy, stddev):
            # pylint: disable=unused-argument
            print(f"iterations {nfev}: energy: {energy}")

        solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP(), callback=callback)
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)
        for idx, line in enumerate(out.getvalue().split("\n")):
            if line.strip():
                self.assertTrue(line.startswith(f"iterations {idx+1}: energy: "))

    def test_vqe_ucc_custom(self):
        """Test custom ansatz in Factory use case"""
        solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_aux_ops_reusability(self):
        """Test that the auxiliary operators can be reused"""
        # Regression test against #1475
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)

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
        solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)

        # now we decide that we want to evaluate another operator
        # for testing simplicity, we just use some pre-constructed auxiliary operators
        _, second_q_ops = self.electronic_structure_problem.second_q_ops()
        aux_ops_dict = self.qubit_converter.convert_match(second_q_ops)
        return calc, res, aux_ops_dict

    def _prepare_uccsd_hf(self, qubit_converter):
        initial_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, qubit_converter)
        ansatz = UCCSD(
            self.num_spatial_orbitals,
            self.num_particles,
            qubit_converter,
            initial_state=initial_state,
        )

        return ansatz

    def test_uccsd_hf(self):
        """uccsd hf test"""
        ansatz = self._prepare_uccsd_hf(self.qubit_converter)

        optimizer = SLSQP(maxiter=100)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)

    @slow_test
    def test_uccsd_hf_qasm(self):
        """uccsd hf test with qasm simulator."""
        qubit_converter = QubitConverter(ParityMapper())
        ansatz = self._prepare_uccsd_hf(qubit_converter)

        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=Estimator(),
            initial_point=[0.0] * ansatz.num_parameters,
        )

        gsc = GroundStateEigensolver(qubit_converter, solver)

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
        qubit_converter = QubitConverter(
            ParityMapper(),
            two_qubit_reduction=True,
            z2symmetry_reduction="auto",
        )

        solver = NumPyMinimumEigensolverFactory()
        gsc = GroundStateEigensolver(qubit_converter, solver)

        result = gsc.solve(problem)
        self.assertAlmostEqual(result.total_energies[0], -7.882, places=2)

    def test_total_dipole(self):
        """Regression test against #198.

        An issue with calculating the dipole moment that had division None/float.
        """
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_dipole_moment_in_debye[0], 0.0, places=1)

    def test_print_result(self):
        """Regression test against #198 and general issues with printing results."""
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            print(res)
        # do NOT change the below! Lines have been truncated as to not force exact numerical matches
        expected = """\
            === GROUND STATE ENERGY ===

            * Electronic ground state energy (Hartree): -1.857
              - computed part:      -1.857
            ~ Nuclear repulsion energy (Hartree): 0.719
            > Total ground state energy (Hartree): -1.137

            === MEASURED OBSERVABLES ===

              0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000

            === DIPOLE MOMENTS ===

            ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.38

              0:
              * Electronic dipole moment (a.u.): [0.0  0.0  1.38
                - computed part:      [0.0  0.0  1.38
              > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.
                             (debye): [0.0  0.0  0.0]  Total: 0.
        """
        for truth, expected in zip(out.getvalue().split("\n"), expected.split("\n")):
            assert truth.strip().startswith(expected.strip())

    def test_default_initial_point(self):
        """Test when using the default initial point."""

        solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)

        np.testing.assert_array_equal(solver.initial_point.to_numpy_array(), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_ucc_factory_with_user_initial_point(self):
        """Test VQEUCCFactory when using it with a user defined initial point."""

        initial_point = np.asarray([1.28074029e-19, 5.92226076e-08, 1.11762559e-01])
        solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP(maxiter=1), initial_point=initial_point)
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        np.testing.assert_array_almost_equal(res.raw_result.optimal_point, initial_point)

    def test_vqe_ucc_factory_with_mp2(self):
        """Test when using MP2InitialPoint to generate the initial point."""

        initial_point = MP2InitialPoint()

        solver = VQEUCCFactory(Estimator(), UCCSD(), SLSQP(), initial_point=initial_point)
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)

        np.testing.assert_array_almost_equal(
            solver.initial_point.to_numpy_array(), self.mp2_initial_point
        )
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_ucc_factory_with_reps(self):
        """Test when using the default initial point with repeated evolved operators."""
        ansatz = UCCSD(
            qubit_converter=self.qubit_converter,
            num_particles=self.num_particles,
            num_spatial_orbitals=self.num_spatial_orbitals,
            reps=2,
        )

        solver = VQEUCCFactory(Estimator(), ansatz, SLSQP())
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)

        np.testing.assert_array_almost_equal(
            solver.initial_point.to_numpy_array(), np.zeros(6, dtype=float)
        )
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_ucc_factory_with_mp2_with_reps(self):
        """Test when using MP2InitialPoint to generate the initial point with repeated evolved
        operators.
        """

        initial_point = MP2InitialPoint()

        ansatz = UCCSD(
            qubit_converter=self.qubit_converter,
            num_particles=self.num_particles,
            num_spatial_orbitals=self.num_spatial_orbitals,
            reps=2,
        )

        solver = VQEUCCFactory(Estimator(), ansatz, SLSQP(), initial_point=initial_point)
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)

        np.testing.assert_array_almost_equal(
            solver.initial_point.to_numpy_array(), np.tile(self.mp2_initial_point, 2)
        )
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)


if __name__ == "__main__":
    unittest.main()
