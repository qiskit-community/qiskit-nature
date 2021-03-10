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

""" Test Qubit Converter """

import unittest

from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from test import QiskitNatureTestCase

from qiskit.opflow import X, Y, Z, I

from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.problems.second_quantization.molecular import fermionic_op_builder


class TestQubitConverter(QiskitNatureTestCase):
    """ Test Qubit Converter """

    REF_H2 = \
        - 0.81054798160031430 * (I ^ I ^ I ^ I) \
        - 0.22575349071287365 * (Z ^ I ^ I ^ I) \
        + 0.17218393211855787 * (I ^ Z ^ I ^ I) \
        + 0.12091263243164174 * (Z ^ Z ^ I ^ I) \
        - 0.22575349071287362 * (I ^ I ^ Z ^ I) \
        + 0.17464343053355980 * (Z ^ I ^ Z ^ I) \
        + 0.16614543242281926 * (I ^ Z ^ Z ^ I) \
        + 0.17218393211855818 * (I ^ I ^ I ^ Z) \
        + 0.16614543242281926 * (Z ^ I ^ I ^ Z) \
        + 0.16892753854646372 * (I ^ Z ^ I ^ Z) \
        + 0.12091263243164174 * (I ^ I ^ Z ^ Z) \
        + 0.04523279999117751 * (X ^ X ^ X ^ X) \
        + 0.04523279999117751 * (Y ^ Y ^ X ^ X) \
        + 0.04523279999117751 * (X ^ X ^ Y ^ Y) \
        + 0.04523279999117751 * (Y ^ Y ^ Y ^ Y)

    def test_mapping_basic(self):
        """ Test mapping to qubit operator """
        driver = HDF5Driver(hdf5_input=self.get_resource_path('test_driver_hdf5.hdf5',
                                                              'drivers/hdf5d'))
        q_molecule = driver.run()
        second_q_ops = []
        fermionic_op = fermionic_op_builder.build_fermionic_op(q_molecule)
        second_q_ops.append(SecondQuantizedOp([fermionic_op]))
        jw_mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(jw_mapper)
        qubit_ops = qubit_conv.to_qubit_ops(second_q_ops)

        self.assertEqual(len(qubit_ops), 1)

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2)


if __name__ == '__main__':
    unittest.main()
