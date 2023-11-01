# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests Hopping Operators builder."""

from test import QiskitNatureTestCase

import unittest

from qiskit_algorithms.utils import algorithm_globals

from qiskit_nature.second_q.algorithms.excited_states_solvers.qeom_vibrational_ops_builder import (
    build_vibrational_ops,
)
from qiskit_nature.second_q.formats.watson import WatsonHamiltonian
from qiskit_nature.second_q.formats.watson_translator import watson_to_problem
from qiskit_nature.second_q.mappers import DirectMapper, TaperedQubitMapper
from qiskit_nature.second_q.problems import HarmonicBasis
import qiskit_nature.optionals as _optionals

from .resources.expected_qeom_ops import (
    expected_hopping_operators_vibrational,
    expected_commutativies_vibrational,
    expected_indices_vibrational,
)


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8

        self.mapper = DirectMapper()
        self.tapered_mapper = TaperedQubitMapper(self.mapper)

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

        self.basis = HarmonicBasis([2, 2])
        self.vibrational_problem = watson_to_problem(watson, self.basis)

    def test_build_hopping_operators_mapper(self):
        """Tests that the correct hopping operator is built with a qubit mapper."""

        hopping_operators, commutativities, indices = build_vibrational_ops(
            self.basis.num_modals, "sd", self.mapper
        )

        with self.subTest("hopping operators"):
            self.assertEqual(
                hopping_operators.keys(), expected_hopping_operators_vibrational.keys()
            )
            for key, exp_key in zip(
                hopping_operators.keys(), expected_hopping_operators_vibrational.keys()
            ):
                self.assertEqual(key, exp_key)
                val = hopping_operators[key]
                exp_val = expected_hopping_operators_vibrational[exp_key]
                if not val.equiv(exp_val):
                    print(val)
                    print(exp_val)
                self.assertTrue(val.equiv(exp_val), msg=(val, exp_val))

        with self.subTest("commutativities"):
            self.assertEqual(commutativities, expected_commutativies_vibrational)

        with self.subTest("excitation indices"):
            self.assertEqual(indices, expected_indices_vibrational)

    def test_build_hopping_operators_taperedmapper(self):
        """Tests that the correct hopping operator is built with a qubit mapper."""

        hopping_operators, commutativities, indices = build_vibrational_ops(
            self.basis.num_modals, "sd", self.tapered_mapper
        )

        with self.subTest("hopping operators"):
            self.assertEqual(
                hopping_operators.keys(), expected_hopping_operators_vibrational.keys()
            )
            for key, exp_key in zip(
                hopping_operators.keys(), expected_hopping_operators_vibrational.keys()
            ):
                self.assertEqual(key, exp_key)
                val = hopping_operators[key]
                exp_val = expected_hopping_operators_vibrational[exp_key]
                if not val.equiv(exp_val):
                    print(val)
                    print(exp_val)
                self.assertTrue(val.equiv(exp_val), msg=(val, exp_val))

        with self.subTest("commutativities"):
            self.assertEqual(commutativities, expected_commutativies_vibrational)

        with self.subTest("excitation indices"):
            self.assertEqual(indices, expected_indices_vibrational)


if __name__ == "__main__":
    unittest.main()
