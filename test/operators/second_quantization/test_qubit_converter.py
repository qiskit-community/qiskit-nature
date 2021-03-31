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
from typing import Optional, List

from test import QiskitNatureTestCase

from qiskit.opflow import X, Y, Z, I, PauliSumOp, Z2Symmetries

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.problems.second_quantization.electronic.builders import fermionic_op_builder
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem

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

    REF_H2_PARITY_2Q_REDUCED_TAPER = \
        - 1.04109314222921250 * I \
        - 0.79587484566286300 * Z \
        - 0.18093119996470994 * X

    def setUp(self):
        super().setUp()
        driver = HDF5Driver(hdf5_input=self.get_resource_path('test_driver_hdf5.hdf5',
                                                              'drivers/hdf5d'))
        self.molecule = driver.run()
        self.num_particles = (self.molecule.num_alpha, self.molecule.num_beta)
        self.h2_op = fermionic_op_builder._build_fermionic_op(self.molecule)

    def test_mapping_basic(self):
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

        with self.subTest('convert_match()'):
            qubit_op = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)

        with self.subTest('Re-use with different mapper'):
            qubit_conv.mapper = ParityMapper()
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)

        with self.subTest('Set two qubit reduction - no effect without num particles'):
            qubit_conv.two_qubit_reduction = True
            qubit_op = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)

        with self.subTest('Force match set num particles'):
            qubit_conv.force_match(self.num_particles)
            qubit_op = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)

    def test_two_qubit_reduction(self):
        """ Test mapping to qubit operator with two qubit reduction """
        mapper = ParityMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)

        with self.subTest('Two qubit reduction ignored as no num particles given'):
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
            self.assertIsNone(qubit_conv.num_particles)

        with self.subTest('Two qubit reduction, num particles given'):
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
            self.assertEqual(qubit_conv.num_particles, self.num_particles)

        with self.subTest('convert_match()'):
            qubit_op = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
            self.assertEqual(qubit_conv.num_particles, self.num_particles)

        with self.subTest('State is reset (Num particles lost)'):
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
            self.assertIsNone(qubit_conv.num_particles)

        with self.subTest('Num particles given again'):
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)

        with self.subTest('Set for no two qubit reduction'):
            qubit_conv.two_qubit_reduction = False
            self.assertFalse(qubit_conv.two_qubit_reduction)
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)

    def test_z2_symmetry(self):
        """ Test mapping to qubit operator with z2 symmetry tapering """
        z2_sector = [-1, 1, -1]

        def finder(z2_symmetries: Z2Symmetries) -> Optional[List[int]]:
            return z2_sector if not z2_symmetries.is_empty() else None

        def find_none(_z2_symmetries: Z2Symmetries) -> Optional[List[int]]:
            return None

        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper)

        with self.subTest('Locator returns None, should be untapered operator'):
            qubit_op = qubit_conv.convert(self.h2_op, sector_locator=find_none)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)

        qubit_op = qubit_conv.convert(self.h2_op, sector_locator=finder)
        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)

        with self.subTest('convert_match()'):
            qubit_op = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)
            self.assertIsNone(qubit_conv.num_particles)
            self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)

    def test_two_qubit_reduction_and_z2_symmetry(self):
        """ Test mapping to qubit operator with z2 symmetry tapering and two qubit reduction """
        z2_sector = [-1]

        def finder(z2_symmetries: Z2Symmetries) -> Optional[List[int]]:
            return z2_sector if not z2_symmetries.is_empty() else None

        mapper = ParityMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)
        qubit_op = qubit_conv.convert(self.h2_op, self.num_particles, sector_locator=finder)
        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPER)
        self.assertEqual(qubit_conv.num_particles, self.num_particles)
        self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)

        with self.subTest('convert_match()'):
            qubit_op = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPER)
            self.assertEqual(qubit_conv.num_particles, self.num_particles)
            self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)

        with self.subTest('Change setting'):
            qubit_conv.z2symmetry_reduction = [1]
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            self.assertNotEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPER)
            qubit_conv.z2symmetry_reduction = [-1]
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPER)

        with self.subTest('Specify sector upfront'):
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True,
                                        z2symmetry_reduction=z2_sector)
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED_TAPER)

        with self.subTest('Specify sector upfront, but invalid content'):
            with self.assertRaises(ValueError):
                _ = QubitConverter(mapper, two_qubit_reduction=True, z2symmetry_reduction=[5])

        with self.subTest('Specify sector upfront, but invalid length'):
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True,
                                        z2symmetry_reduction=[-1, 1])
            with self.assertRaises(QiskitNatureError):
                _ = qubit_conv.convert(self.h2_op, self.num_particles)

    def test_molecular_problem_sector_locator_z2_symmetry(self):
        """ Test mapping to qubit operator with z2 symmetry tapering and two qubit reduction """

        driver = HDF5Driver(hdf5_input=self.get_resource_path('test_driver_hdf5.hdf5',
                                                              'drivers/hdf5d'))
        problem = ElectronicStructureProblem(driver)

        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)
        qubit_op = qubit_conv.convert(problem.second_q_ops()[0], self.num_particles,
                                      sector_locator=problem.symmetry_sector_locator)
        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)

if __name__ == '__main__':
    unittest.main()
