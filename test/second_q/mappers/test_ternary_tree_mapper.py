# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Ternary Tree Mapper """

import unittest
from test import QiskitNatureTestCase

import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyEigensolver

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import TernaryTreeMapper
from qiskit_nature.second_q.operators import MajoranaOp


class TestTernaryTreeMapper(QiskitNatureTestCase):
    """Test Ternary Tree Mapper"""

    REF_H2 = SparsePauliOp.from_list(
        [
            ("IIII", -0.81054798160031430),
            ("ZIII", -0.22575349071287365),
            ("IZII", +0.17218393211855787),
            ("ZZII", +0.12091263243164174),
            ("IIZI", -0.22575349071287362),
            ("ZIZI", +0.17464343053355980),
            ("IZZI", +0.16614543242281926),
            ("IIIZ", +0.17218393211855818),
            ("ZIIZ", +0.16614543242281926),
            ("IZIZ", +0.16892753854646372),
            ("IIZZ", +0.12091263243164174),
            ("XXXX", +0.04523279999117751),
            ("YYXX", +0.04523279999117751),
            ("XXYY", +0.04523279999117751),
            ("YYYY", +0.04523279999117751),
        ]
    )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_mapping(self):
        """Test spectrum of H2 molecule."""
        driver = PySCFDriver()
        driver_result = driver.run()
        fermionic_op, _ = driver_result.second_q_ops()
        majorana_op = MajoranaOp.from_fermionic_op(fermionic_op)
        mapper = TernaryTreeMapper()
        qubit_op = mapper.map(majorana_op)
        result = NumPyEigensolver().compute_eigenvalues(qubit_op).eigenvalues
        expected = NumPyEigensolver().compute_eigenvalues(TestTernaryTreeMapper.REF_H2).eigenvalues
        self.assertTrue(np.isclose(result, expected))

    def test_pauli_table(self):
        """Test that Pauli table satisfies Majorana anticommutation relations."""
        for num_modes in range(1, 10):
            pauli_tab = []
            for pauli in TernaryTreeMapper("XYZ").pauli_table(num_modes):
                self.assertEqualSparsePauliOp(pauli[1], SparsePauliOp([""]))
                pauli_tab.append(pauli[0] / 2)
            self.assertEqual(len(pauli_tab), num_modes)
            identity = SparsePauliOp("I" * pauli_tab[0].num_qubits)
            for i, first in enumerate(pauli_tab):
                anticommutator = (first.dot(first) + first.dot(first)).simplify()
                self.assertEqual(anticommutator, 2 * identity)
                for j in range(i):
                    second = pauli_tab[j]
                    anticommutator = (first.dot(second) + second.dot(first)).simplify()
                    self.assertEqual(anticommutator, 0 * identity)

    def test_mapping_for_single_op(self):
        """Test for double register operator."""
        with self.subTest("test all ops for num_modes=4"):
            num_modes = 4
            ops = ["_0", "_1", "_2", "_3"]
            expected = ["IY", "IZ", "XX", "YX"]
            for op, pauli_string in zip(ops, expected):
                majorana_op = MajoranaOp({op: 1}, num_modes=num_modes)
                expected_op = SparsePauliOp.from_list([(pauli_string, 1)])
                mapped = TernaryTreeMapper("XYZ").map(majorana_op)
                self.assertEqualSparsePauliOp(mapped, expected_op)

        with self.subTest("test all ops for num_modes=8"):
            num_modes = 8
            ops = ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7"]
            expected = ["IIXX", "IIYX", "IIZX", "IXIY", "IYIY", "IZIY", "XIIZ", "YIIZ"]
            for op, pauli_string in zip(ops, expected):
                majorana_op = MajoranaOp({op: 1}, num_modes=num_modes)
                expected_op = SparsePauliOp.from_list([(pauli_string, 1)])
                mapped = TernaryTreeMapper("XYZ").map(majorana_op)
                self.assertEqualSparsePauliOp(mapped, expected_op)

        with self.subTest("test parameters"):
            a = Parameter("a")
            op = MajoranaOp({"_0": a}, num_modes=2)
            expected = SparsePauliOp.from_list([("X", a)], dtype=object)
            qubit_op = TernaryTreeMapper("XYZ").map(op)
            self.assertEqual(qubit_op, expected)

        with self.subTest("test empty operator"):
            op = MajoranaOp({}, num_modes=2)
            expected = SparsePauliOp.from_list([("I", 0)])
            qubit_op = TernaryTreeMapper("XYZ").map(op)
            self.assertEqual(qubit_op, expected)

        with self.subTest("test constant operator"):
            op = MajoranaOp({"": 2.2}, num_modes=2)
            expected = SparsePauliOp.from_list([("I", 2.2)])
            qubit_op = TernaryTreeMapper("XYZ").map(op)
            self.assertEqual(qubit_op, expected)

    def test_mapping_for_list_ops(self):
        """Test for list of single register operator."""
        ops = [
            MajoranaOp({"_0": 1}, num_modes=2),
            MajoranaOp({"_1": 1}, num_modes=2),
            MajoranaOp({"_0 _1": 1}, num_modes=2),
        ]
        expected = [
            SparsePauliOp.from_list([("X", 1)]),
            SparsePauliOp.from_list([("Y", 1)]),
            SparsePauliOp.from_list([("Z", 1j)]),
        ]

        mapped_ops = TernaryTreeMapper("XYZ").map(ops)
        self.assertEqual(len(mapped_ops), len(expected))
        for mapped_op, expected_op in zip(mapped_ops, expected):
            self.assertEqual(mapped_op, expected_op)

    def test_mapping_for_dict_ops(self):
        """Test for dict of single register operator."""
        ops = {
            "gamma0": MajoranaOp({"_0": 1}, num_modes=2),
            "gamma1": MajoranaOp({"_1": 1}, num_modes=2),
            "gamma0 gamma1": MajoranaOp({"_0 _1": 1}, num_modes=2),
        }
        expected = {
            "gamma0": SparsePauliOp.from_list([("X", 1)]),
            "gamma1": SparsePauliOp.from_list([("Y", 1)]),
            "gamma0 gamma1": SparsePauliOp.from_list([("Z", 1j)]),
        }

        mapped_ops = TernaryTreeMapper("XYZ").map(ops)
        self.assertEqual(len(mapped_ops), len(expected))
        for k in mapped_ops.keys():
            self.assertEqual(mapped_ops[k], expected[k])

    def test_mapping_overwrite_reg_len(self):
        """Test overwriting the register length."""
        op = MajoranaOp({"_0 _1": 1}, num_modes=2)
        expected = MajoranaOp({"_0 _1": 1}, num_modes=3)
        mapper = TernaryTreeMapper("XYZ")
        self.assertEqual(mapper.map(op, register_length=3), mapper.map(expected))

    def test_mapping_pauli_priority(self):
        """Test different settings for Pauli priority."""
        ops = [MajoranaOp({"_0": 1}, num_modes=2), MajoranaOp({"_1": 1}, num_modes=2)]
        strings = "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
        for string in strings:
            with self.subTest(string):
                mapped = TernaryTreeMapper(string).map(ops)
                expected = [
                    SparsePauliOp.from_list([(string[0], 1)]),
                    SparsePauliOp.from_list([(string[1], 1)]),
                ]
                self.assertEqual(mapped, expected)

        with self.subTest("invalid"):
            with self.assertRaises(ValueError):
                TernaryTreeMapper("ABC")


if __name__ == "__main__":
    unittest.main()
