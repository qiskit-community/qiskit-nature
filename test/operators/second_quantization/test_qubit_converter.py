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

from test import QiskitNatureTestCase

from qiskit.opflow import X, Y, Z, I, PauliSumOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.molecular import fermionic_op_builder


class TestQubitConverter(QiskitNatureTestCase):
    """ Test Qubit Converter """

    REF_H2_JW = \
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

    REF_H2_PARITY = \
        - 0.81054798160031430 * (I ^ I ^ I ^ I) \
        - 0.22575349071287365 * (Z ^ Z ^ I ^ I) \
        + 0.12091263243164174 * (I ^ I ^ Z ^ I) \
        + 0.12091263243164174 * (Z ^ I ^ Z ^ I) \
        + 0.17218393211855787 * (I ^ Z ^ Z ^ I) \
        + 0.17218393211855818 * (I ^ I ^ I ^ Z) \
        + 0.16614543242281926 * (I ^ Z ^ I ^ Z) \
        + 0.16614543242281926 * (Z ^ Z ^ I ^ Z) \
        - 0.22575349071287362 * (I ^ I ^ Z ^ Z) \
        + 0.16892753854646372 * (I ^ Z ^ Z ^ Z) \
        + 0.17464343053355980 * (Z ^ Z ^ Z ^ Z) \
        + 0.04523279999117751 * (I ^ X ^ I ^ X) \
        + 0.04523279999117751 * (Z ^ X ^ I ^ X) \
        - 0.04523279999117751 * (I ^ X ^ Z ^ X) \
        - 0.04523279999117751 * (Z ^ X ^ Z ^ X)

    REF_H2_PARITY_2Q_REDUCED = \
        - 1.05237324646359750 * (I ^ I) \
        - 0.39793742283143140 * (Z ^ I) \
        + 0.39793742283143163 * (I ^ Z) \
        - 0.01128010423438501 * (Z ^ Z) \
        + 0.18093119996471000 * (X ^ X)

    REF_H2_JW_TAPERED = \
        - 1.04109314222921270 * I \
        - 0.79587484566286240 * Z \
        + 0.18093119996470988 * X

    REF_H2_PARITY_2Q_REDUCED_TAPERED = \
        - 1.04109314222921250 * I \
        - 0.79587484566286300 * Z \
        - 0.18093119996470994 * X

    def setUp(self):
        super().setUp()
        driver = HDF5Driver(hdf5_input=self.get_resource_path('test_driver_hdf5.hdf5',
                                                              'drivers/hdf5d'))
        self.molecule = driver.run()
        self.num_particles = (self.molecule.num_alpha, self.molecule.num_beta)
        self.h2_op = fermionic_op_builder.build_fermionic_op(self.molecule)

    def test_mapping_basic(self):
        """ Test mapping to qubit operator """
        second_q_ops = [self.h2_op]
        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper)
        qubit_ops = qubit_conv.convert(second_q_ops)

        self.assertEqual(len(qubit_ops), 1)

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_JW)

    def test_mapping_basic_nolist(self):
        """ Test mapping to qubit operator """
        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper)
        qubit_op = qubit_conv.convert(self.h2_op)

        self.assertIsInstance(qubit_op, PauliSumOp)

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)

        with self.subTest('Re-use test'):
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)

        with self.subTest('convert_more()'):
            qubit_op = qubit_conv.convert_more(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)

    def test_mapping_basic_list(self):
        """ Test mapping to qubit operator of a list > 1 """
        # We create 3 of the same operator as a simple/quick test
        second_q_ops = [self.h2_op] * 3
        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper)
        qubit_ops = qubit_conv.convert(second_q_ops)

        self.assertEqual(len(qubit_ops), 3)

        # Since we created 3 identical operators (the qubit converter should not be
        # doing any optimization in that regard) lets make sure they were converted
        # independently by checking they are different instance via their ids.
        self.assertNotEqual(id(qubit_ops[0]), id(qubit_ops[1]))
        self.assertNotEqual(id(qubit_ops[0]), id(qubit_ops[2]))
        self.assertNotEqual(id(qubit_ops[1]), id(qubit_ops[2]))

        self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_JW)
        self.assertEqual(qubit_ops[1], TestQubitConverter.REF_H2_JW)
        self.assertEqual(qubit_ops[2], TestQubitConverter.REF_H2_JW)

    def test_two_qubit_reduction(self):
        """ Test mapping to qubit operator """
        second_q_ops = [self.h2_op]
        mapper = ParityMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)

        with self.subTest('Two qubit reduction ignored as no num particles given'):
            qubit_ops = qubit_conv.convert(second_q_ops)
            self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_PARITY)
            self.assertFalse(qubit_conv.did_two_qubit_reduction)
            self.assertIsNone(qubit_conv.num_particles)

        with self.subTest('Two qubit reduction, num particles given'):
            qubit_ops = qubit_conv.convert(second_q_ops, self.num_particles)
            self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
            self.assertTrue(qubit_conv.did_two_qubit_reduction)
            self.assertEqual(qubit_conv.num_particles, self.num_particles)

            with self.subTest('convert_more()'):
                qubit_ops = qubit_conv.convert_more(second_q_ops)
                self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
                self.assertTrue(qubit_conv.did_two_qubit_reduction)
                self.assertEqual(qubit_conv.num_particles, self.num_particles)

            with self.subTest('Change setting: properties and convert_more() should raise error'):
                qubit_conv.two_qubit_reduction = True  # Will reset state
                self.assertTrue(qubit_conv.two_qubit_reduction)
                with self.assertRaises(QiskitNatureError):
                    _ = qubit_conv.did_two_qubit_reduction
                with self.assertRaises(QiskitNatureError):
                    _ = qubit_conv.num_particles
                with self.assertRaises(QiskitNatureError):
                    _ = qubit_conv.convert_more(second_q_ops)

            with self.subTest('State is reset (Num particles lost)'):
                qubit_ops = qubit_conv.convert(second_q_ops)
                self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_PARITY)
                self.assertIsNone(qubit_conv.num_particles)

            with self.subTest('Num particles given again'):
                qubit_ops = qubit_conv.convert(second_q_ops, self.num_particles)
                self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)

            with self.subTest('Set for no two qubit reduction'):
                qubit_conv.two_qubit_reduction = False
                self.assertFalse(qubit_conv.two_qubit_reduction)
                qubit_ops = qubit_conv.convert(second_q_ops)
                self.assertEqual(qubit_ops[0], TestQubitConverter.REF_H2_PARITY)

    def test_z2_symmetry(self):
        z2_sector = [-1, 1, -1]
        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper)
        qubit_op = qubit_conv.convert(self.h2_op, z2symmetry_reduction=z2_sector)
        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)

        with self.subTest('convert_more()'):
            qubit_op = qubit_conv.convert_more(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)
            self.assertFalse(qubit_conv.did_two_qubit_reduction)
            self.assertIsNone(qubit_conv.num_particles)
            self.assertListEqual(qubit_conv.z2_symmetries.tapering_values, z2_sector)

        with self.subTest('Change setting: properties and convert_more() should raise error'):
            qubit_conv.z2symmetry_reduction = [1, 1, 1]  # Will reset state
            with self.assertRaises(QiskitNatureError):
                self.assertListEqual(qubit_conv.z2_symmetries.tapering_values, [])
            with self.assertRaises(QiskitNatureError):
                _ = qubit_conv.convert_more(self.h2_op)

    def test_two_qubit_reduction_and_z2_symmetry(self):
        z2_sector = [-1]
        mapper = ParityMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)
        qubit_op = qubit_conv.convert(self.h2_op, self.num_particles, z2_sector)
        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPERED)
        self.assertTrue(qubit_conv.did_two_qubit_reduction)
        self.assertEqual(qubit_conv.num_particles, self.num_particles)
        self.assertListEqual(qubit_conv.z2_symmetries.tapering_values, z2_sector)

        with self.subTest('convert_more()'):
            qubit_op = qubit_conv.convert_more(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPERED)
            self.assertTrue(qubit_conv.did_two_qubit_reduction)
            self.assertEqual(qubit_conv.num_particles, self.num_particles)
            self.assertListEqual(qubit_conv.z2_symmetries.tapering_values, z2_sector)

        with self.subTest('Change setting: properties and convert_more() should raise error'):
            qubit_conv.z2symmetry_reduction = [1]  # Will reset state
            with self.assertRaises(QiskitNatureError):
                self.assertListEqual(qubit_conv.z2_symmetries.tapering_values, [])

        with self.subTest('Specify sector upfront'):
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True,
                                        z2symmetry_reduction=z2_sector)
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPERED)

        with self.subTest('Specify sector upfront, but invalid content'):
            with self.assertRaises(ValueError):
                _ = QubitConverter(mapper, two_qubit_reduction=True, z2symmetry_reduction=[5])

        with self.subTest('Specify sector upfront, but invalid length'):
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True,
                                        z2symmetry_reduction=[-1, 1])
            with self.assertRaises(QiskitNatureError):
                _ = qubit_conv.convert(self.h2_op, self.num_particles)


if __name__ == '__main__':
    unittest.main()
