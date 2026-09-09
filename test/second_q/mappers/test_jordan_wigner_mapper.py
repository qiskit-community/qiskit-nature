# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
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

    h2_fermionic_op: FermionicOp | None = None

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

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if _optionals.HAS_PYSCF:
            driver_result = PySCFDriver().run()
            cls.h2_fermionic_op, _ = driver_result.second_q_ops()

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_mapping(self):
        """Test mapping to qubit operator"""
        assert self.h2_fermionic_op is not None
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(self.h2_fermionic_op)
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

    def test_reverse_map_single_paulis(self):
        """Test reverse mapping of single-qubit Pauli operators."""
        test_cases = {
            "I": FermionicOp({"": 1}, num_spin_orbitals=1),
            "X": FermionicOp({"+_0": 1, "-_0": 1}, num_spin_orbitals=1),
            "Y": FermionicOp({"+_0": 1j, "-_0": -1j}, num_spin_orbitals=1),
            "Z": FermionicOp({"": 1, "+_0 -_0": -2}, num_spin_orbitals=1),
        }

        mapper = JordanWignerMapper()
        for label, expected in test_cases.items():
            with self.subTest(label):
                qubit_op = SparsePauliOp.from_list([(label, 1)])
                actual = mapper.reverse_map(qubit_op)
                self.assertTrue(expected.equiv(actual))
                self.assertEqual(actual.num_spin_orbitals, qubit_op.num_qubits)

    def test_reverse_map_zero_and_identity(self):
        """Test reverse mapping preserves the register length for zero and identity."""
        test_cases = {
            "zero": (
                SparsePauliOp.from_list([], num_qubits=3),
                FermionicOp({}, num_spin_orbitals=3),
            ),
            "identity": (
                SparsePauliOp.from_list([("III", 1)]),
                FermionicOp({"": 1}, num_spin_orbitals=3),
            ),
        }

        mapper = JordanWignerMapper()
        for name, (qubit_op, expected) in test_cases.items():
            with self.subTest(name):
                actual = mapper.reverse_map(qubit_op)
                self.assertTrue(expected.equiv(actual))
                self.assertEqual(actual.num_spin_orbitals, qubit_op.num_qubits)

    def test_reverse_map_phases(self):
        """Test phase handling while reverse mapping multi-qubit Pauli operators."""
        mapper = JordanWignerMapper()
        test_cases = (
            ("XY", 1),
            ("XX", 0.25 - 0.75j),
            ("YX", -0.5j),
            ("XXXX", 1),
        )
        for label, coefficient in test_cases:
            with self.subTest(label):
                qubit_op = SparsePauliOp.from_list([(label, coefficient)])
                recovered_fermionic_op = mapper.reverse_map(qubit_op)
                self.assertTrue(qubit_op.equiv(mapper.map(recovered_fermionic_op)))

    def test_reverse_map_fermionic_round_trip(self):
        """Test round trips of small fermionic operators."""
        test_cases = {
            "creation": FermionicOp({"+_0": 1}, num_spin_orbitals=1),
            "orbital order": FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2),
        }

        mapper = JordanWignerMapper()
        for name, fermionic_op in test_cases.items():
            with self.subTest(name):
                recovered_fermionic_op = mapper.reverse_map(mapper.map(fermionic_op))
                self.assertTrue(fermionic_op.normal_order().equiv(recovered_fermionic_op))
                self.assertEqual(
                    recovered_fermionic_op.num_spin_orbitals,
                    fermionic_op.num_spin_orbitals,
                )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_reverse_map_h2(self):
        """Test the reverse-mapping round trip of an H2 Hamiltonian."""
        assert self.h2_fermionic_op is not None
        mapper = JordanWignerMapper()
        recovered_fermionic_op = mapper.reverse_map(mapper.map(self.h2_fermionic_op))
        self.assertTrue(self.h2_fermionic_op.normal_order().equiv(recovered_fermionic_op))


if __name__ == "__main__":
    unittest.main()
