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

"""Test the UVCC Ansatz."""

from test import QiskitNatureTestCase
from test.circuit.library.ansatzes.utils.vibrational_op_label_creator import _create_labels

import unittest

from ddt import ddt, data, unpack

from qiskit import BasicAer, transpile
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit_nature.circuit.library import UVCC, VSCF
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.second_quantization.operators import VibrationalOp
from qiskit_nature.converters.second_quantization import QubitConverter


def assert_ucc_like_ansatz(test_case, ansatz, num_modals, expected_ops):
    """Assertion utility."""
    excitation_ops = ansatz.excitation_ops()

    test_case.assertEqual(len(excitation_ops), len(expected_ops))
    for op, exp in zip(excitation_ops, expected_ops):
        test_case.assertListEqual(op.to_list(), exp.to_list())

    ansatz._build()
    test_case.assertEqual(ansatz.num_qubits, sum(num_modals))


@ddt
class TestUVCC(QiskitNatureTestCase):
    """Tests for the UVCC Ansatz."""

    @unpack
    @data(
        ("s", [2], [VibrationalOp([("+-", 1j), ("-+", -1j)], 1, 2)]),
        (
            "s",
            [2, 2],
            [
                VibrationalOp([("+-II", 1j), ("-+II", -1j)], 2, 2),
                VibrationalOp([("II+-", 1j), ("II-+", -1j)], 2, 2),
            ],
        ),
    )
    def test_ucc_ansatz(self, excitations, num_modals, expect):
        """Tests the UVCC Ansatz."""
        converter = QubitConverter(DirectMapper())

        ansatz = UVCC(qubit_converter=converter, num_modals=num_modals, excitations=excitations)

        assert_ucc_like_ansatz(self, ansatz, num_modals, expect)

    def test_transpile_no_parameters(self):
        """Test transpilation without parameters"""

        qubit_converter = QubitConverter(mapper=DirectMapper())

        ansatz = UVCC(qubit_converter=qubit_converter, num_modals=[2], excitations="s")
        ansatz = transpile(ansatz, optimization_level=3)
        self.assertEqual(ansatz.num_qubits, 2)


class TestUVCCVSCF(QiskitNatureTestCase):
    """Test for these extensions."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.reference_energy = 592.5346633819712

    def test_uvcc_vscf(self):
        """uvcc vscf test"""

        co2_2modes_2modals_2body = [
            [
                [[[0, 0, 0]], 320.8467332810141],
                [[[0, 1, 1]], 1760.878530705873],
                [[[1, 0, 0]], 342.8218290247543],
                [[[1, 1, 1]], 1032.396323618631],
            ],
            [
                [[[0, 0, 0], [1, 0, 0]], -57.34003649795117],
                [[[0, 0, 1], [1, 0, 0]], -56.33205925807966],
                [[[0, 1, 0], [1, 0, 0]], -56.33205925807966],
                [[[0, 1, 1], [1, 0, 0]], -60.13032761856809],
                [[[0, 0, 0], [1, 0, 1]], -65.09576309934431],
                [[[0, 0, 1], [1, 0, 1]], -62.2363839133389],
                [[[0, 1, 0], [1, 0, 1]], -62.2363839133389],
                [[[0, 1, 1], [1, 0, 1]], -121.5533969109279],
                [[[0, 0, 0], [1, 1, 0]], -65.09576309934431],
                [[[0, 0, 1], [1, 1, 0]], -62.2363839133389],
                [[[0, 1, 0], [1, 1, 0]], -62.2363839133389],
                [[[0, 1, 1], [1, 1, 0]], -121.5533969109279],
                [[[0, 0, 0], [1, 1, 1]], -170.744837386338],
                [[[0, 0, 1], [1, 1, 1]], -167.7433236025723],
                [[[0, 1, 0], [1, 1, 1]], -167.7433236025723],
                [[[0, 1, 1], [1, 1, 1]], -179.0536532281924],
            ],
        ]
        num_modes = 2
        num_modals = [2, 2]

        vibrational_op_labels = _create_labels(co2_2modes_2modals_2body)
        vibr_op = VibrationalOp(vibrational_op_labels, num_modes, num_modals)

        converter = QubitConverter(DirectMapper())

        qubit_op = converter.convert_match(vibr_op)

        init_state = VSCF(num_modals)

        uvcc_ansatz = UVCC(converter, num_modals, "sd", initial_state=init_state)

        q_instance = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            seed_transpiler=90,
            seed_simulator=12,
        )
        optimizer = COBYLA(maxiter=1000)

        algo = VQE(uvcc_ansatz, optimizer=optimizer, quantum_instance=q_instance)
        vqe_result = algo.compute_minimum_eigenvalue(qubit_op)

        energy = vqe_result.optimal_value

        self.assertAlmostEqual(energy, self.reference_energy, places=4)

    def test_build_uvcc(self):
        """Test building UVCC"""
        uvcc = UVCC()

        with self.subTest("Check defaulted construction"):
            self.assertIsNone(uvcc.num_modals)
            self.assertIsNone(uvcc.excitations)
            self.assertIsNone(uvcc.qubit_converter)
            self.assertIsNone(uvcc.operators)
            self.assertIsNone(uvcc.excitation_list)
            self.assertEqual(uvcc.num_qubits, 0)
            with self.assertRaises(ValueError):
                _ = uvcc.data

        with self.subTest("Set num modals"):
            uvcc.num_modals = [2, 2]
            self.assertListEqual(uvcc.num_modals, [2, 2])
            self.assertIsNone(uvcc.operators)
            with self.assertRaises(ValueError):
                _ = uvcc.data

        with self.subTest("Set excitations"):
            uvcc.excitations = "sd"
            self.assertEqual(uvcc.excitations, "sd")
            self.assertIsNone(uvcc.operators)
            with self.assertRaises(ValueError):
                _ = uvcc.data

        with self.subTest("Set qubit converter to complete build"):
            converter = QubitConverter(DirectMapper())
            uvcc.qubit_converter = converter
            self.assertEqual(uvcc.qubit_converter, converter)
            self.assertIsNotNone(uvcc.operators)
            self.assertEqual(len(uvcc.operators), 3)
            self.assertEqual(uvcc.num_qubits, 4)
            self.assertIsNotNone(uvcc.data)

        with self.subTest("Set custom operators"):
            self.assertEqual(len(uvcc.operators), 3)
            uvcc.operators = uvcc.operators[:2]
            self.assertEqual(len(uvcc.operators), 2)
            self.assertEqual(uvcc.num_qubits, 4)

        with self.subTest("Reset operators back to as per UVCC"):
            uvcc.operators = None
            self.assertEqual(uvcc.num_qubits, 4)
            self.assertIsNotNone(uvcc.operators)
            self.assertEqual(len(uvcc.operators), 3)

        with self.subTest("Set num modals differently"):
            uvcc.num_modals = [3, 3]
            self.assertEqual(uvcc.num_modals, [3, 3])
            self.assertIsNotNone(uvcc.operators)
            self.assertEqual(len(uvcc.operators), 8)

        with self.subTest("Change excitations"):
            uvcc.excitations = "s"
            self.assertIsNotNone(uvcc.operators)
            self.assertEqual(len(uvcc.operators), 4)


if __name__ == "__main__":
    unittest.main()
