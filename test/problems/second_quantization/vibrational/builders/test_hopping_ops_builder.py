# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
from test.algorithms.excited_state_solvers.test_bosonic_esc_calculation import (
    _DummyBosonicDriver,
)
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals

from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import VibrationalStructureProblem
from qiskit_nature.problems.second_quantization.vibrational.builders.hopping_ops_builder import (
    _build_qeom_hopping_ops,
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

        self.vibrational_problem = VibrationalStructureProblem(
            self.driver, self.basis_size, self.truncation_order
        )

        self.qubit_converter = QubitConverter(DirectMapper())
        self.vibrational_problem.second_q_ops()
        self.watson_hamiltonian = self.vibrational_problem.properties_transformed
        self.num_modals = [self.basis_size] * self.watson_hamiltonian.num_modes

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built from QMolecule."""
        # TODO extract it somewhere
        expected_hopping_operators = (
            {
                "E_0": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, False, False, False, False, False, False],
                            [True, True, False, False, False, True, False, False],
                            [True, True, False, False, True, False, False, False],
                            [True, True, False, False, True, True, False, False],
                        ],
                        coeffs=[0.25 + 0.0j, 0.0 - 0.25j, 0.0 + 0.25j, 0.25 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "Edag_0": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, False, False, False, False, False, False],
                            [True, True, False, False, False, True, False, False],
                            [True, True, False, False, True, False, False, False],
                            [True, True, False, False, True, True, False, False],
                        ],
                        coeffs=[0.25 + 0.0j, 0.0 + 0.25j, 0.0 - 0.25j, 0.25 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "E_1": PauliSumOp(
                    SparsePauliOp(
                        [
                            [False, False, True, True, False, False, False, False],
                            [False, False, True, True, False, False, False, True],
                            [False, False, True, True, False, False, True, False],
                            [False, False, True, True, False, False, True, True],
                        ],
                        coeffs=[0.25 + 0.0j, 0.0 - 0.25j, 0.0 + 0.25j, 0.25 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "Edag_1": PauliSumOp(
                    SparsePauliOp(
                        [
                            [False, False, True, True, False, False, False, False],
                            [False, False, True, True, False, False, False, True],
                            [False, False, True, True, False, False, True, False],
                            [False, False, True, True, False, False, True, True],
                        ],
                        coeffs=[0.25 + 0.0j, 0.0 + 0.25j, 0.0 - 0.25j, 0.25 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "E_2": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, True, True, False, False, False, False],
                            [True, True, True, True, False, False, False, True],
                            [True, True, True, True, False, False, True, False],
                            [True, True, True, True, False, False, True, True],
                            [True, True, True, True, False, True, False, False],
                            [True, True, True, True, False, True, False, True],
                            [True, True, True, True, False, True, True, False],
                            [True, True, True, True, False, True, True, True],
                            [True, True, True, True, True, False, False, False],
                            [True, True, True, True, True, False, False, True],
                            [True, True, True, True, True, False, True, False],
                            [True, True, True, True, True, False, True, True],
                            [True, True, True, True, True, True, False, False],
                            [True, True, True, True, True, True, False, True],
                            [True, True, True, True, True, True, True, False],
                            [True, True, True, True, True, True, True, True],
                        ],
                        coeffs=[
                            0.0625 + 0.0j,
                            0.0 - 0.0625j,
                            0.0 + 0.0625j,
                            0.0625 + 0.0j,
                            0.0 - 0.0625j,
                            -0.0625 + 0.0j,
                            0.0625 + 0.0j,
                            0.0 - 0.0625j,
                            0.0 + 0.0625j,
                            0.0625 + 0.0j,
                            -0.0625 + 0.0j,
                            0.0 + 0.0625j,
                            0.0625 + 0.0j,
                            0.0 - 0.0625j,
                            0.0 + 0.0625j,
                            0.0625 + 0.0j,
                        ],
                    ),
                    coeff=1.0,
                ),
                "Edag_2": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, True, True, False, False, False, False],
                            [True, True, True, True, False, False, False, True],
                            [True, True, True, True, False, False, True, False],
                            [True, True, True, True, False, False, True, True],
                            [True, True, True, True, False, True, False, False],
                            [True, True, True, True, False, True, False, True],
                            [True, True, True, True, False, True, True, False],
                            [True, True, True, True, False, True, True, True],
                            [True, True, True, True, True, False, False, False],
                            [True, True, True, True, True, False, False, True],
                            [True, True, True, True, True, False, True, False],
                            [True, True, True, True, True, False, True, True],
                            [True, True, True, True, True, True, False, False],
                            [True, True, True, True, True, True, False, True],
                            [True, True, True, True, True, True, True, False],
                            [True, True, True, True, True, True, True, True],
                        ],
                        coeffs=[
                            0.0625 + 0.0j,
                            0.0 + 0.0625j,
                            0.0 - 0.0625j,
                            0.0625 + 0.0j,
                            0.0 + 0.0625j,
                            -0.0625 + 0.0j,
                            0.0625 + 0.0j,
                            0.0 + 0.0625j,
                            0.0 - 0.0625j,
                            0.0625 + 0.0j,
                            -0.0625 + 0.0j,
                            0.0 - 0.0625j,
                            0.0625 + 0.0j,
                            0.0 + 0.0625j,
                            0.0 - 0.0625j,
                            0.0625 + 0.0j,
                        ],
                    ),
                    coeff=1.0,
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

        hopping_operators = _build_qeom_hopping_ops(self.num_modals, self.qubit_converter)
        self.assertEqual(hopping_operators, expected_hopping_operators)
