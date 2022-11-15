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
from test.second_q.algorithms.excited_state_solvers.test_bosonic_esc_calculation import (
    _DummyBosonicDriver,
)

from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals

from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import DirectMapper
from qiskit_nature.second_q.algorithms.excited_states_solvers.qeom_vibrational_ops_builder import (
    build_vibrational_ops,
)
from .resources.expected_qeom_ops import (
    expected_hopping_operators_vibrational,
    expected_commutativies_vibrational,
    expected_indices_vibrational,
)


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.driver = _DummyBosonicDriver()
        self.qubit_converter = QubitConverter(DirectMapper())
        self.basis_size = 2
        self.truncation_order = 2

        self.vibrational_problem = self.driver.run()
        self.vibrational_problem._num_modals = self.basis_size
        self.vibrational_problem.truncation_order = self.truncation_order

        self.qubit_converter = QubitConverter(DirectMapper())
        self.vibrational_problem.second_q_ops()
        self.grouped_property_transformed = self.vibrational_problem
        self.num_modals = [self.basis_size] * self.vibrational_problem.num_modes

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built."""
        # TODO extract it somewhere

        hopping_operators, commutativities, indices = build_vibrational_ops(
            self.num_modals, self.qubit_converter
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
                if not val.equals(exp_val):
                    print(val)
                    print(exp_val)
                self.assertTrue(val.equals(exp_val), msg=(val, exp_val))

        with self.subTest("commutativities"):
            self.assertEqual(commutativities, expected_commutativies_vibrational)

        with self.subTest("excitation indices"):
            self.assertEqual(indices, expected_indices_vibrational)
