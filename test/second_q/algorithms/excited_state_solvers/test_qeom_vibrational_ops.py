# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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

from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals

from qiskit_nature.second_q.algorithms.excited_states_solvers.qeom_vibrational_ops_builder import (
    build_vibrational_ops,
)
from qiskit_nature.second_q.formats.watson import WatsonHamiltonian
from qiskit_nature.second_q.formats.watson_translator import watson_to_problem
from qiskit_nature.second_q.mappers import DirectMapper, QubitConverter
from qiskit_nature.second_q.problems import HarmonicBasis
import qiskit_nature.optionals as _optionals


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
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

        self.basis = HarmonicBasis([2, 2])
        self.vibrational_problem = watson_to_problem(watson, self.basis)

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built."""
        # TODO extract it somewhere
        expected_hopping_operators = (
            {
                "E_0": PauliSumOp.from_list(
                    [("IIXX", 0.25), ("IIYX", 0.25j), ("IIXY", -0.25j), ("IIYY", 0.25)]
                ),
                "Edag_0": PauliSumOp.from_list(
                    [("IIXX", 0.25), ("IIYX", -0.25j), ("IIXY", 0.25j), ("IIYY", 0.25)]
                ),
                "E_1": PauliSumOp.from_list(
                    [("XXII", 0.25), ("YXII", 0.25j), ("XYII", -0.25j), ("YYII", 0.25)]
                ),
                "Edag_1": PauliSumOp.from_list(
                    [("XXII", 0.25), ("YXII", -0.25j), ("XYII", 0.25j), ("YYII", 0.25)]
                ),
                "E_2": PauliSumOp.from_list(
                    [
                        ("XXXX", 0.0625),
                        ("YXXX", 0.0625j),
                        ("XYXX", -0.0625j),
                        ("YYXX", 0.0625),
                        ("XXYX", 0.0625j),
                        ("YXYX", -0.0625),
                        ("XYYX", 0.0625),
                        ("YYYX", 0.0625j),
                        ("XXXY", -0.0625j),
                        ("YXXY", 0.0625),
                        ("XYXY", -0.0625),
                        ("YYXY", -0.0625j),
                        ("XXYY", 0.0625),
                        ("YXYY", 0.0625j),
                        ("XYYY", -0.0625j),
                        ("YYYY", 0.0625),
                    ]
                ),
                "Edag_2": PauliSumOp.from_list(
                    [
                        ("XXXX", 0.0625),
                        ("YXXX", -0.0625j),
                        ("XYXX", 0.0625j),
                        ("YYXX", 0.0625),
                        ("XXYX", -0.0625j),
                        ("YXYX", -0.0625),
                        ("XYYX", 0.0625),
                        ("YYYX", -0.0625j),
                        ("XXXY", 0.0625j),
                        ("YXXY", 0.0625),
                        ("XYXY", -0.0625),
                        ("YYXY", 0.0625j),
                        ("XXYY", 0.0625),
                        ("YXYY", -0.0625j),
                        ("XYYY", 0.0625j),
                        ("YYYY", 0.0625),
                    ]
                ),
            },
            {},
            {
                "E_0": ((0,), (1,)),
                "Edag_0": ((1,), (0,)),
                "E_1": ((2,), (3,)),
                "Edag_1": ((3,), (2,)),
                "E_2": ((0, 2), (1, 3)),
                "Edag_2": ((1, 3), (0, 2)),
            },
        )

        hopping_operators = build_vibrational_ops(self.basis.num_modals, "sd", self.qubit_converter)
        self.assertEqual(hopping_operators, expected_hopping_operators)


if __name__ == "__main__":
    unittest.main()
