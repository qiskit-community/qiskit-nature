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

""" Test Qubit Converter """

import contextlib
import io
import unittest
import warnings
from test import QiskitNatureTestCase
from typing import List, Optional

from qiskit.opflow import I, PauliSumOp, X, Y, Z, Z2Symmetries
from qiskit.quantum_info import SparsePauliOp

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    QubitConverter,
)
from qiskit_nature.settings import settings


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
class TestQubitConverter(QiskitNatureTestCase):
    """Test Qubit Converter"""

    REF_H2_JW = (
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

    REF_H2_PARITY = (
        -0.81054798160031430 * (I ^ I ^ I ^ I)
        - 0.22575349071287365 * (Z ^ Z ^ I ^ I)
        + 0.12091263243164174 * (I ^ I ^ Z ^ I)
        + 0.12091263243164174 * (Z ^ I ^ Z ^ I)
        + 0.17218393211855787 * (I ^ Z ^ Z ^ I)
        + 0.17218393211855818 * (I ^ I ^ I ^ Z)
        + 0.16614543242281926 * (I ^ Z ^ I ^ Z)
        + 0.16614543242281926 * (Z ^ Z ^ I ^ Z)
        - 0.22575349071287362 * (I ^ I ^ Z ^ Z)
        + 0.16892753854646372 * (I ^ Z ^ Z ^ Z)
        + 0.17464343053355980 * (Z ^ Z ^ Z ^ Z)
        + 0.04523279999117751 * (I ^ X ^ I ^ X)
        + 0.04523279999117751 * (Z ^ X ^ I ^ X)
        - 0.04523279999117751 * (I ^ X ^ Z ^ X)
        - 0.04523279999117751 * (Z ^ X ^ Z ^ X)
    )

    REF_H2_PARITY_2Q_REDUCED = (
        -1.05237324646359750 * (I ^ I)
        - 0.39793742283143140 * (Z ^ I)
        + 0.39793742283143163 * (I ^ Z)
        - 0.01128010423438501 * (Z ^ Z)
        + 0.18093119996471000 * (X ^ X)
    )

    REF_H2_JW_TAPERED = -1.04109314222921270 * I - 0.79587484566286240 * Z + 0.18093119996470988 * X

    REF_H2_PARITY_TAPERED = (
        -1.04109314222921250 * I - 0.79587484566286300 * Z - 0.18093119996470994 * X
    )

    def setUp(self):
        super().setUp()
        driver = PySCFDriver()
        self.driver_result = driver.run()
        self.num_particles = self.driver_result.num_particles  # (1, 1)
        self.h2_op, _ = self.driver_result.second_q_ops()
        self.mapper = ParityMapper()
        self.qubit_conv = QubitConverter(self.mapper)

    def test_mapping_basic(self):
        """Test mapping to qubit operator"""
        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper)

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        aux = settings.use_pauli_sum_op
        try:
            settings.use_pauli_sum_op = True
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertIsInstance(qubit_op, PauliSumOp)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)
            settings.use_pauli_sum_op = False
            qubit_op = qubit_conv.convert(self.h2_op)
            self.assertIsInstance(qubit_op, SparsePauliOp)
            self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_JW.primitive)
        finally:
            settings.use_pauli_sum_op = aux

        with self.subTest("Re-use test"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(self.h2_op)
                self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_JW.primitive)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("convert_match()"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_JW.primitive)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("Re-use with different mapper"):
            qubit_conv.mapper = ParityMapper()
            qubit_conv.two_qubit_reduction = True
            qubit_op = qubit_conv.convert(self.h2_op)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)

        with self.subTest("Force match set num particles"):
            qubit_conv.force_match(num_particles=self.num_particles)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED.primitive
                )
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("Convert with number of particles"):
            qubit_conv.force_match(num_particles=None)
            qubit_op = qubit_conv.convert(self.h2_op, num_particles=self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)

    def test_two_qubit_reduction(self):
        """Test mapping to qubit operator with two qubit reduction"""
        mapper = ParityMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)

        with self.subTest("Two qubit reduction produces list as no particle number is given"):
            qubit_op = qubit_conv.convert(self.h2_op)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
            self.assertIsNone(qubit_conv.num_particles)

        with self.subTest("Two qubit reduction, num particles given"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED.primitive
                )
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("convert_match()"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED.primitive
                )
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("State is reset (Num particles lost)"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
                self.assertIsNone(qubit_conv.num_particles)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(self.h2_op)
                self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_PARITY.primitive)
                self.assertIsNone(qubit_conv.num_particles)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("Num particles given again"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED.primitive
                )
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("Set two qubit reduction to False"):
            qubit_conv.two_qubit_reduction = False
            self.assertFalse(qubit_conv.two_qubit_reduction)
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(self.h2_op)
                self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_PARITY.primitive)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("Set two qubit reduction to False, set particle number in convert"):
            qubit_conv.two_qubit_reduction = False
            self.assertFalse(qubit_conv.two_qubit_reduction)
            qubit_op = qubit_conv.convert(self.h2_op, num_particles=self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)

        with self.subTest("Set two qubit reduction to False, set particle numbers in the mapper"):
            qubit_conv.two_qubit_reduction = False
            self.assertFalse(qubit_conv.two_qubit_reduction)
            qubit_conv.force_match(num_particles=self.num_particles)
            qubit_op = qubit_conv.convert(self.h2_op)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)

        # Regression test against https://github.com/Qiskit/qiskit-nature/issues/271
        with self.subTest("Two qubit reduction skipped when operator too small"):
            qubit_conv.two_qubit_reduction = True
            small_op = FermionicOp({"+_0 -_0": 1.0, "-_1 +_1": 1.0}, num_spin_orbitals=2)
            expected_op = 1.0 * (I ^ I) - 0.5 * (I ^ Z) + 0.5 * (Z ^ Z)
            with contextlib.redirect_stderr(io.StringIO()) as out:
                qubit_op = qubit_conv.convert(small_op, num_particles=self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, expected_op)
            self.assertTrue(
                out.getvalue()
                .strip()
                .startswith(
                    "The original qubit operator only contains 2 qubits! "
                    "Skipping the requested two-qubit reduction!"
                )
            )

    def test_paritymapper_two_qubit_reduction(self):
        """Test mapping to qubit operator with two qubit reduction from the parity Mapper."""

        with self.subTest("No particle number in the mapper and the convert method"):
            mapper = ParityMapper()
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=False)
            qubit_op = qubit_conv.convert(self.h2_op)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
            self.assertIsNone(qubit_conv.num_particles)

        with self.subTest("Set particle number in the mapper only"):
            mapper = ParityMapper(num_particles=(1, 1))
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=False)
            qubit_op = qubit_conv.convert(self.h2_op)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
            self.assertIsNone(qubit_conv.num_particles)

        with self.subTest("Two qubit reduction is False and num particles given"):
            mapper = ParityMapper(num_particles=(1, 1))
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=False)
            qubit_op = qubit_conv.convert(self.h2_op, num_particles=(1, 1))
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY)
            self.assertIsNone(mapper.num_particles)

        with self.subTest("Set particle number in the converter only"):
            mapper = ParityMapper()
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)
            qubit_op = qubit_conv.convert(self.h2_op, num_particles=(1, 1))
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_2Q_REDUCED)
            self.assertEqual(qubit_conv.num_particles, self.num_particles)

        # Regression test against https://github.com/Qiskit/qiskit-nature/issues/271
        with self.subTest("Two qubit reduction skipped when operator too small"):
            mapper = ParityMapper(num_particles=self.num_particles)
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)
            small_op = FermionicOp({"+_0 -_0": 1.0, "-_1 +_1": 1.0}, num_spin_orbitals=2)
            expected_op = 1.0 * (I ^ I) - 0.5 * (I ^ Z) + 0.5 * (Z ^ Z)
            with contextlib.redirect_stderr(io.StringIO()) as out:
                qubit_op = qubit_conv.convert(small_op, num_particles=self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, expected_op)
            self.assertTrue(
                out.getvalue()
                .strip()
                .startswith(
                    "The original qubit operator only contains 2 qubits! "
                    "Skipping the requested two-qubit reduction!"
                )
            )

    def test_z2_symmetry(self):
        """Test mapping to qubit operator with z2 symmetry tapering"""
        z2_sector = [-1, 1, -1]

        def cb_finder(
            z2_symmetries: Z2Symmetries, converter: QubitConverter
        ) -> Optional[List[int]]:
            return z2_sector if not z2_symmetries.is_empty() else None

        def cb_find_none(
            _z2_symmetries: Z2Symmetries, converter: QubitConverter
        ) -> Optional[List[int]]:
            return None

        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper, z2symmetry_reduction="auto")

        with self.subTest("Locator returns None, should be untapered operator"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert(self.h2_op, sector_locator=cb_find_none)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(self.h2_op, sector_locator=cb_find_none)
                self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_JW.primitive)
            finally:
                settings.use_pauli_sum_op = aux

        aux = settings.use_pauli_sum_op
        try:
            settings.use_pauli_sum_op = True
            qubit_op = qubit_conv.convert(self.h2_op, sector_locator=cb_finder)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)
            settings.use_pauli_sum_op = False
            qubit_op = qubit_conv.convert(self.h2_op, sector_locator=cb_finder)
            self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED.primitive)
        finally:
            settings.use_pauli_sum_op = aux

        with self.subTest("convert_match()"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)
                self.assertIsNone(qubit_conv.num_particles)
                self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_JW_TAPERED.primitive
                )
                self.assertIsNone(qubit_conv.num_particles)
                self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)
            finally:
                settings.use_pauli_sum_op = aux

    def test_two_qubit_reduction_and_z2_symmetry(self):
        """Test mapping to qubit operator with z2 symmetry tapering and two qubit reduction"""
        z2_sector = [-1]

        def cb_finder(
            z2_symmetries: Z2Symmetries, converter: QubitConverter
        ) -> Optional[List[int]]:
            return z2_sector if not z2_symmetries.is_empty() else None

        mapper = ParityMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True, z2symmetry_reduction="auto")
        qubit_op = qubit_conv.convert(self.h2_op, self.num_particles, sector_locator=cb_finder)
        if not isinstance(qubit_op, PauliSumOp):
            qubit_op = PauliSumOp(qubit_op)
        self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED)
        self.assertEqual(qubit_conv.num_particles, self.num_particles)
        self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)

        with self.subTest("convert_match()"):
            aux = settings.use_pauli_sum_op
            try:
                qubit_op = qubit_conv.convert_match(self.h2_op)
                if not isinstance(qubit_op, PauliSumOp):
                    qubit_op = PauliSumOp(qubit_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED)
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
                self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert(
                    self.h2_op, self.num_particles, sector_locator=cb_finder
                )
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED.primitive
                )
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
                self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("convert_match()"):
            aux = settings.use_pauli_sum_op
            try:
                settings.use_pauli_sum_op = True
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED)
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
                self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)
                settings.use_pauli_sum_op = False
                qubit_op = qubit_conv.convert_match(self.h2_op)
                self.assertEqualSparsePauliOp(
                    qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED.primitive
                )
                self.assertEqual(qubit_conv.num_particles, self.num_particles)
                self.assertListEqual(qubit_conv.z2symmetries.tapering_values, z2_sector)
            finally:
                settings.use_pauli_sum_op = aux

        with self.subTest("Change setting"):
            qubit_conv.z2symmetry_reduction = [1]
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertNotEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED)
            qubit_conv.z2symmetry_reduction = [-1]
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED)

        with self.subTest("Specify sector upfront"):
            qubit_conv = QubitConverter(
                mapper, two_qubit_reduction=True, z2symmetry_reduction=z2_sector
            )
            qubit_op = qubit_conv.convert(self.h2_op, self.num_particles)
            if not isinstance(qubit_op, PauliSumOp):
                qubit_op = PauliSumOp(qubit_op)
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_PARITY_TAPERED)

        with self.subTest("Specify sector upfront, but invalid content"):
            with self.assertRaises(ValueError):
                _ = QubitConverter(mapper, two_qubit_reduction=True, z2symmetry_reduction=[5])

        with self.subTest("Specify sector upfront, but invalid length"):
            qubit_conv = QubitConverter(
                mapper, two_qubit_reduction=True, z2symmetry_reduction=[-1, 1]
            )
            with self.assertRaises(QiskitNatureError):
                _ = qubit_conv.convert(self.h2_op, self.num_particles)

    def test_molecular_problem_sector_locator_z2_symmetry(self):
        """Test mapping to qubit operator with z2 symmetry tapering and two qubit reduction"""

        driver = PySCFDriver()
        problem = driver.run()

        mapper = JordanWignerMapper()
        qubit_conv = QubitConverter(mapper, two_qubit_reduction=True, z2symmetry_reduction="auto")
        main_op, _ = problem.second_q_ops()
        aux = settings.use_pauli_sum_op
        try:
            settings.use_pauli_sum_op = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                qubit_op = qubit_conv.convert(
                    main_op,
                    self.num_particles,
                    sector_locator=problem.symmetry_sector_locator,
                )
            self.assertEqual(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED)
            settings.use_pauli_sum_op = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                qubit_op = qubit_conv.convert(
                    main_op,
                    self.num_particles,
                    sector_locator=problem.symmetry_sector_locator,
                )
            self.assertEqualSparsePauliOp(qubit_op, TestQubitConverter.REF_H2_JW_TAPERED.primitive)
        finally:
            settings.use_pauli_sum_op = aux

    def test_compatibiliy_with_mappers(self):
        """Test that Qubit converter and mappers produces the same results."""

        with self.subTest("JordanWigner Mapper"):
            mapper = JordanWignerMapper()
            qubit_conv = QubitConverter(mapper)
            qubit_op_converter = mapper.map(self.h2_op)
            qubit_op_mapper = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op_converter, qubit_op_mapper)

        with self.subTest("Parity Mapper"):
            mapper = ParityMapper()
            qubit_conv = QubitConverter(mapper)
            qubit_op_converter = mapper.map(self.h2_op)
            qubit_op_mapper = qubit_conv.convert(self.h2_op)
            self.assertEqual(qubit_op_converter, qubit_op_mapper)

        with self.subTest("Parity Mapper and two qubit reduction"):
            mapper = ParityMapper(num_particles=(1, 1))
            qubit_conv = QubitConverter(mapper, two_qubit_reduction=True)
            qubit_op_converter = mapper.map(self.h2_op)
            qubit_op_mapper = qubit_conv.convert_match(self.h2_op)
            self.assertEqual(qubit_op_converter, qubit_op_mapper)

    def test_error_with_tapered_qubit_mapper(self):
        """Test that the qubit converter cannot be used with a Tapered Qubit Mapper"""

        mapper = JordanWignerMapper()
        tq_mapper = self.driver_result.get_tapered_mapper(mapper)
        with self.assertRaises(ValueError):
            _ = QubitConverter(tq_mapper)


if __name__ == "__main__":
    unittest.main()
