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

""" Test Numerical qEOM excited states calculation """

import contextlib
import io
import unittest

from test import QiskitNatureTestCase

import numpy as np

from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals

from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
    VQEUVCCFactory,
    QEOM,
    ExcitedStatesEigensolver,
    NumPyEigensolverFactory,
)
from qiskit_nature.second_q.circuit.library import UVCCSD
from qiskit_nature.second_q.formats.watson import WatsonHamiltonian
from qiskit_nature.second_q.formats.watson_translator import watson_to_problem
from qiskit_nature.second_q.mappers import DirectMapper, QubitConverter
from qiskit_nature.second_q.problems import HarmonicBasis
import qiskit_nature.optionals as _optionals


class TestBosonicESCCalculation(QiskitNatureTestCase):
    """Test Numerical QEOM excited states calculation"""

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.reference_energies = [
            1889.95738428,
            3294.21806197,
            4287.26821341,
            5819.76975784,
        ]

        self.qubit_converter = QubitConverter(DirectMapper())

        import sparse as sp  # pylint: disable=import-error

        watson = WatsonHamiltonian(
            quadratic_force_constants=sp.as_coo(
                {
                    (0, 0): 605.3643675,
                    (1, 1): 340.5950575,
                },
                shape=(2, 2),
            ),
            cubic_force_constants=sp.as_coo(
                {
                    (1, 0, 0): -89.09086530649508,
                    (1, 1, 1): -15.590557244410897,
                },
                shape=(2, 2, 2),
            ),
            quartic_force_constants=sp.as_coo(
                {
                    (0, 0, 0, 0): 1.6512647916666667,
                    (1, 1, 0, 0): 5.03965375,
                    (1, 1, 1, 1): 0.43840625000000005,
                },
                shape=(2, 2, 2, 2),
            ),
            kinetic_coefficients=sp.as_coo(
                {
                    (0, 0): -605.3643675,
                    (1, 1): -340.5950575,
                },
                shape=(2, 2),
            ),
        )

        basis = HarmonicBasis([2, 2])
        self.vibrational_problem = watson_to_problem(watson, basis)

    def _assert_energies(self, computed, references, *, places=4):
        with self.subTest("same number of energies"):
            self.assertEqual(len(computed), len(references))

        with self.subTest("ground state"):
            self.assertAlmostEqual(computed[0], references[0], places=places)

        for i in range(1, len(computed)):
            with self.subTest(f"{i}. excited state"):
                self.assertAlmostEqual(computed[i], references[i], places=places)

    def test_numpy_mes(self):
        """Test with NumPyMinimumEigensolver"""
        estimator = Estimator()
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, estimator, "sd")
        results = esc.solve(self.vibrational_problem)
        self._assert_energies(results.computed_vibrational_energies, self.reference_energies)

    def test_numpy_factory(self):
        """Test with NumPyEigensolver"""
        solver = NumPyEigensolverFactory(use_default_filter_criterion=True)
        esc = ExcitedStatesEigensolver(self.qubit_converter, solver)
        results = esc.solve(self.vibrational_problem)
        self._assert_energies(results.computed_vibrational_energies, self.reference_energies)

    def test_vqe_uvccsd_factory(self):
        """Test with VQE plus UVCCSD"""
        optimizer = COBYLA(maxiter=5000)
        estimator = Estimator()
        solver = VQEUVCCFactory(estimator, UVCCSD(), optimizer)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, estimator, "sd")
        results = esc.solve(self.vibrational_problem)
        self._assert_energies(
            results.computed_vibrational_energies, self.reference_energies, places=0
        )

    def test_vqe_uvcc_factory_with_user_initial_point(self):
        """Test VQEUVCCFactory when using it with a user defined initial point."""
        initial_point = np.asarray([-7.35250290e-05, -9.73079292e-02, -5.43346282e-05])
        estimator = Estimator()
        optimizer = COBYLA(maxiter=1)
        solver = VQEUVCCFactory(estimator, UVCCSD(), optimizer, initial_point=initial_point)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, estimator, "sd")
        results = esc.solve(self.vibrational_problem)
        np.testing.assert_array_almost_equal(
            results.raw_result.ground_state_raw_result.optimal_point, initial_point
        )

    def test_vqe_uvccsd_with_callback(self):
        """Test VQE UVCCSD with callback."""

        def cb_callback(nfev, parameters, energy, stddev):
            print(f"iterations {nfev}: energy: {energy}")

        estimator = Estimator()
        optimizer = COBYLA(maxiter=5000)

        solver = VQEUVCCFactory(estimator, UVCCSD(), optimizer, callback=cb_callback)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, estimator, "sd")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            results = esc.solve(self.vibrational_problem)
        self._assert_energies(
            results.computed_vibrational_energies, self.reference_energies, places=0
        )
        for idx, line in enumerate(out.getvalue().split("\n")):
            if line.strip():
                self.assertTrue(line.startswith(f"iterations {idx+1}: energy: "))


if __name__ == "__main__":
    unittest.main()
