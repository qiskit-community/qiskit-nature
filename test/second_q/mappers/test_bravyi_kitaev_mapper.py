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

""" Test Bravyi-Kitaev Mapper """

import unittest
from test import QiskitNatureTestCase

from qiskit.quantum_info import SparsePauliOp

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_nature.second_q.operators import FermionicOp


class TestBravyiKitaevMapper(QiskitNatureTestCase):
    """Test Bravyi-Kitaev Mapper"""

    REF_H2 = SparsePauliOp.from_list(
        [
            ("IIII", -0.81054798160031430),
            ("IZII", +0.17218393211855787),
            ("IIZI", +0.12091263243164174),
            ("ZIZI", +0.12091263243164174),
            ("ZZZI", -0.22575349071287365),
            ("IIIZ", +0.17218393211855818),
            ("IZIZ", +0.16892753854646372),
            ("ZZIZ", +0.17464343053355980),
            ("IIZZ", -0.22575349071287362),
            ("IZZZ", +0.16614543242281926),
            ("ZZZZ", +0.16614543242281926),
            ("IXIX", +0.04523279999117751),
            ("ZXIX", +0.04523279999117751),
            ("IXZX", -0.04523279999117751),
            ("ZXZX", -0.04523279999117751),
        ]
    )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_mapping(self):
        """Test mapping to qubit operator"""
        driver = PySCFDriver()
        driver_result = driver.run()
        fermionic_op, _ = driver_result.second_q_ops()
        mapper = BravyiKitaevMapper()
        qubit_op = mapper.map(fermionic_op)
        self.assertTrue(qubit_op.equiv(TestBravyiKitaevMapper.REF_H2))

    def test_mapping_for_single_op(self):
        """Test for single register operator."""
        with self.subTest("test +"):
            op = FermionicOp({"+_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])
            self.assertEqualSparsePauliOp(BravyiKitaevMapper().map(op), expected)

        with self.subTest("test -"):
            op = FermionicOp({"-_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)])
            self.assertEqualSparsePauliOp(BravyiKitaevMapper().map(op), expected)

        with self.subTest("test N"):
            op = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])
            self.assertEqualSparsePauliOp(BravyiKitaevMapper().map(op), expected)

        with self.subTest("test E"):
            op = FermionicOp({"-_0 +_0": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])
            self.assertEqualSparsePauliOp(BravyiKitaevMapper().map(op), expected)

        with self.subTest("test I"):
            op = FermionicOp({"": 1}, num_spin_orbitals=1)
            expected = SparsePauliOp.from_list([("I", 1)])
            self.assertEqualSparsePauliOp(BravyiKitaevMapper().map(op), expected)

    def test_mapping_overwrite_reg_len(self):
        """Test overwriting the register length."""
        op = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
        expected = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=3)
        mapper = BravyiKitaevMapper()
        self.assertEqual(mapper.map(op, register_length=3), mapper.map(expected))


if __name__ == "__main__":
    unittest.main()
