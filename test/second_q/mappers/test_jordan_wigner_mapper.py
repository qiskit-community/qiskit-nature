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

""" Test Jordan Wigner Mapper """

import unittest
from test import QiskitNatureTestCase

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


class TestJordanWignerMapper(QiskitNatureTestCase):
    """Test Jordan Wigner Mapper"""

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
        """Test mapping to qubit operator"""
        driver = PySCFDriver()
        driver_result = driver.run()
        fermionic_op, _ = driver_result.second_q_ops()
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermionic_op)
        self.assertTrue(qubit_op.equiv(TestJordanWignerMapper.REF_H2))

    def test_mapping_for_single_op(self):
        """Test for single register operator."""
        with self.subTest("test +"):
            op = FermionicOp({"+_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])
            self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)

        with self.subTest("test -"):
            op = FermionicOp({"-_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)])
            self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)

        with self.subTest("test N"):
            op = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])
            self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)

        with self.subTest("test E"):
            op = FermionicOp({"-_0 +_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])
            self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)

        with self.subTest("test I"):
            op = FermionicOp({"": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("I", 1)])
            self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)

        with self.subTest("test parameters"):
            a = Parameter("a")
            op = FermionicOp({"+_0": a})
            expected = SparsePauliOp.from_list([("X", 0.5 * a), ("Y", -0.5j * a)], dtype=object)
            qubit_op = JordanWignerMapper().map(op)
            self.assertEqual(qubit_op, expected)

    def test_mapping_for_list_ops(self):
        """Test for list of single register operator."""
        ops = [
            FermionicOp({"+_0": 1}, num_spin_orbitals=1),
            FermionicOp({"-_0": 1}, num_spin_orbitals=1),
            FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1),
            FermionicOp({"-_0 +_0": 1}, num_spin_orbitals=1),
            FermionicOp({"": 1}, num_spin_orbitals=1),
        ]
        expected = [
            SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)]),
            SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)]),
            SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)]),
            SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)]),
            SparsePauliOp.from_list([("I", 1)]),
        ]

        mapped_ops = JordanWignerMapper().map(ops)
        self.assertEqual(len(mapped_ops), len(expected))
        for mapped_op, expected_op in zip(mapped_ops, expected):
            self.assertEqual(mapped_op, expected_op)

    def test_mapping_for_dict_ops(self):
        """Test for dict of single register operator."""
        ops = {
            "+": FermionicOp({"+_0": 1}, num_spin_orbitals=1),
            "-": FermionicOp({"-_0": 1}, num_spin_orbitals=1),
            "N": FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1),
            "E": FermionicOp({"-_0 +_0": 1}, num_spin_orbitals=1),
            "I": FermionicOp({"": 1}, num_spin_orbitals=1),
        }
        expected = {
            "+": SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)]),
            "-": SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)]),
            "N": SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)]),
            "E": SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)]),
            "I": SparsePauliOp.from_list([("I", 1)]),
        }

        mapped_ops = JordanWignerMapper().map(ops)
        self.assertEqual(len(mapped_ops), len(expected))
        for k in mapped_ops.keys():
            self.assertEqual(mapped_ops[k], expected[k])

    def test_mapping_overwrite_reg_len(self):
        """Test overwriting the register length."""
        op = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
        expected = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=3)
        mapper = JordanWignerMapper()
        self.assertEqual(mapper.map(op, register_length=3), mapper.map(expected))


if __name__ == "__main__":
    unittest.main()
