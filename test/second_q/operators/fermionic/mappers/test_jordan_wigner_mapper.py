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

""" Test Jordan Wigner Mapper """

import unittest
from test import QiskitNatureTestCase

from qiskit.opflow import I, PauliSumOp, X, Y, Z

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.second_q.operators.fermionic import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


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

    def test_mapping(self):
        """Test mapping to qubit operator"""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("test_driver_hdf5.hdf5", "drivers/second_q/hdf5d")
        )
        driver_result = driver.run()
        fermionic_op = driver_result.second_q_ops()["ElectronicEnergy"]
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermionic_op)

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        self.assertEqual(qubit_op, TestJordanWignerMapper.REF_H2)

    def test_allows_two_qubit_reduction(self):
        """Test this returns False for this mapper"""
        mapper = JordanWignerMapper()
        self.assertFalse(mapper.allows_two_qubit_reduction)

    def test_mapping_for_single_op(self):
        """Test for single register operator."""
        with self.subTest("test +"):
            op = FermionicOp("+", display_format="dense")
            expected = PauliSumOp.from_list([("X", 0.5), ("Y", -0.5j)])
            self.assertEqual(JordanWignerMapper().map(op), expected)

        with self.subTest("test -"):
            op = FermionicOp("-", display_format="dense")
            expected = PauliSumOp.from_list([("X", 0.5), ("Y", 0.5j)])
            self.assertEqual(JordanWignerMapper().map(op), expected)

        with self.subTest("test N"):
            op = FermionicOp("N", display_format="dense")
            expected = PauliSumOp.from_list([("I", 0.5), ("Z", -0.5)])
            self.assertEqual(JordanWignerMapper().map(op), expected)

        with self.subTest("test E"):
            op = FermionicOp("E", display_format="dense")
            expected = PauliSumOp.from_list([("I", 0.5), ("Z", 0.5)])
            self.assertEqual(JordanWignerMapper().map(op), expected)

        with self.subTest("test I"):
            op = FermionicOp("I", display_format="dense")
            expected = PauliSumOp.from_list([("I", 1)])
            self.assertEqual(JordanWignerMapper().map(op), expected)


if __name__ == "__main__":
    unittest.main()
