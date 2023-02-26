# This code is part of Qiskit.
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
from qiskit.opflow import I, PauliSumOp, X, Y, Z
from qiskit.quantum_info import SparsePauliOp

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature import settings


class TestJordanWignerMapper(QiskitNatureTestCase):
    """Test Jordan Wigner Mapper"""

    REF_H2 = (
        -0.81054798160031430 * (I ^ I ^ I ^ I)
        - 0.22575349071287365 * (Z ^ I ^ I ^ I)
        + 0.17218393211855787 * (I ^ Z ^ I ^ I)
        + 0.12091263243164174 * (Z ^ Z ^ I ^ I)
        - 0.22575349071287362 * (I ^ I ^ Z ^ I)
        + 0.17464343053355980 * (Z ^ I ^ Z ^ I)
        + 0.16614543242281926 * (I ^ Z ^ Z ^ I)
        + 0.17218393211855818 * (I ^ I ^ I ^ Z)
        + 0.16614543242281926 * (Z ^ I ^ I ^ Z)
        + 0.16892753854646372 * (I ^ Z ^ I ^ Z)
        + 0.12091263243164174 * (I ^ I ^ Z ^ Z)
        + 0.04523279999117751 * (X ^ X ^ X ^ X)
        + 0.04523279999117751 * (Y ^ Y ^ X ^ X)
        + 0.04523279999117751 * (X ^ X ^ Y ^ Y)
        + 0.04523279999117751 * (Y ^ Y ^ Y ^ Y)
    )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_mapping(self):
        """Test mapping to qubit operator"""
        driver = PySCFDriver()
        driver_result = driver.run()
        fermionic_op, _ = driver_result.second_q_ops()
        mapper = JordanWignerMapper()

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        aux = settings.use_pauli_sum_op
        try:
            settings.use_pauli_sum_op = True
            qubit_op = mapper.map(fermionic_op)
            self.assertEqual(qubit_op, TestJordanWignerMapper.REF_H2)
            settings.use_pauli_sum_op = False
            qubit_op = mapper.map(fermionic_op)
            self.assertEqualSparsePauliOp(qubit_op, TestJordanWignerMapper.REF_H2.primitive)
        finally:
            settings.use_pauli_sum_op = aux

    def test_mapping_for_single_op(self):
        """Test for single register operator."""
        with self.subTest("test +"):
            op = FermionicOp({"+_0": 1}, num_spin_orbitals=1)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                expected = PauliSumOp.from_list([("X", 0.5), ("Y", -0.5j)])
                self.assertEqual(JordanWignerMapper().map(op), expected)
                settings.use_pauli_sum_op = False
                expected = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])
                self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("test -"):
            op = FermionicOp({"-_0": 1}, num_spin_orbitals=1)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                expected = PauliSumOp.from_list([("X", 0.5), ("Y", 0.5j)])
                self.assertEqual(JordanWignerMapper().map(op), expected)
                settings.use_pauli_sum_op = False
                expected = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)])
                self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("test N"):
            op = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                expected = PauliSumOp.from_list([("I", 0.5), ("Z", -0.5)])
                self.assertEqual(JordanWignerMapper().map(op), expected)
                settings.use_pauli_sum_op = False
                expected = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])
                self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("test E"):
            op = FermionicOp({"-_0 +_0": 1}, num_spin_orbitals=1)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                expected = PauliSumOp.from_list([("I", 0.5), ("Z", 0.5)])
                self.assertEqual(JordanWignerMapper().map(op), expected)
                settings.use_pauli_sum_op = False
                expected = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])
                self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("test I"):
            op = FermionicOp({"": 1}, num_spin_orbitals=1)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                expected = PauliSumOp.from_list([("I", 1)])
                self.assertEqual(JordanWignerMapper().map(op), expected)
                settings.use_pauli_sum_op = False
                expected = SparsePauliOp.from_list([("I", 1)])
                self.assertEqualSparsePauliOp(JordanWignerMapper().map(op), expected)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("test parameters"):
            a = Parameter("a")
            op = FermionicOp({"+_0": a})
            expected = SparsePauliOp.from_list([("X", 0.5 * a), ("Y", -0.5j * a)], dtype=object)
            qubit_op = JordanWignerMapper().map(op)
            if isinstance(qubit_op, PauliSumOp):
                qubit_op = qubit_op.primitive
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
            PauliSumOp.from_list([("X", 0.5), ("Y", -0.5j)]),
            PauliSumOp.from_list([("X", 0.5), ("Y", 0.5j)]),
            PauliSumOp.from_list([("I", 0.5), ("Z", -0.5)]),
            PauliSumOp.from_list([("I", 0.5), ("Z", 0.5)]),
            PauliSumOp.from_list([("I", 1)]),
        ]

        mapped_ops = JordanWignerMapper().map(ops)
        self.assertEqual(len(mapped_ops), len(expected))
        for mapped_op, expected_op in zip(mapped_ops, expected):
            if not isinstance(mapped_op, PauliSumOp):
                mapped_op = PauliSumOp(mapped_op)
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
            "+": PauliSumOp.from_list([("X", 0.5), ("Y", -0.5j)]),
            "-": PauliSumOp.from_list([("X", 0.5), ("Y", 0.5j)]),
            "N": PauliSumOp.from_list([("I", 0.5), ("Z", -0.5)]),
            "E": PauliSumOp.from_list([("I", 0.5), ("Z", 0.5)]),
            "I": PauliSumOp.from_list([("I", 1)]),
        }

        mapped_ops = JordanWignerMapper().map(ops)
        self.assertEqual(len(mapped_ops), len(expected))
        for k in mapped_ops.keys():
            if not isinstance(mapped_ops[k], PauliSumOp):
                mapped_ops[k] = PauliSumOp(mapped_ops[k])
            self.assertEqual(mapped_ops[k], expected[k])

    def test_mapping_overwrite_reg_len(self):
        """Test overwriting the register length."""
        op = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
        expected = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=3)
        mapper = JordanWignerMapper()
        self.assertEqual(mapper.map(op, register_length=3), mapper.map(expected))


if __name__ == "__main__":
    unittest.main()
