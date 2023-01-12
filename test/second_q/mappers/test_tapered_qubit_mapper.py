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

import unittest
from test import QiskitNatureTestCase

from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitConverter
from qiskit_nature.second_q.mappers.tapered_qubit_mapper import TaperedQubitMapper


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
class TestTaperedQubitMapper(QiskitNatureTestCase):
    """Test Tapered Qubit Mapper"""

    REF_H2_JW = SparsePauliOp.from_list(
        [
            ("IIII", -0.81054798160031430),
            ("ZIII", -0.22575349071287365),
            ("IZII", 0.17218393211855787),
            ("ZZII", 0.12091263243164174),
            ("IIZI", -0.22575349071287362),
            ("ZIZI", 0.17464343053355980),
            ("IZZI", 0.16614543242281926),
            ("IIIZ", 0.17218393211855818),
            ("ZIIZ", 0.16614543242281926),
            ("IZIZ", 0.16892753854646372),
            ("IIZZ", 0.12091263243164174),
            ("XXXX", 0.04523279999117751),
            ("YYXX", 0.04523279999117751),
            ("XXYY", 0.04523279999117751),
            ("YYYY", 0.04523279999117751),
        ]
    )

    REF_H2_JW_TAPERED_LIST = [
        SparsePauliOp.from_list([("I", 0.10713912), ("Z", -0.10713912)]),
        SparsePauliOp.from_list([("I", -0.80483209), ("Z", -0.45150698)]),
        SparsePauliOp.from_list([("I", -0.81626387), ("Z", 0.34436787)]),
        SparsePauliOp.from_list([("I", -1.06365335), ("X", 0.1809312)]),
        SparsePauliOp.from_list([("I", -0.80483209), ("Z", -0.45150698)]),
        SparsePauliOp.from_list([("I", -1.04109314), ("Z", -0.79587485), ("X", 0.1809312)]),
        SparsePauliOp.from_list([("I", -1.24458455)]),
        SparsePauliOp.from_list([("I", -0.81626387), ("Z", -0.34436787)]),
    ]

    REF_H2_JW_TAPERED = SparsePauliOp.from_list(
        [("I", -1.04109314), ("Z", -0.79587485), ("X", 0.1809312)]
    )

    REF_H2_PARITY = SparsePauliOp.from_list(
        [
            ("IIII", -0.81054798160031430),
            ("ZZII", -0.22575349071287365),
            ("IIZI", +0.12091263243164174),
            ("ZIZI", +0.12091263243164174),
            ("IZZI", +0.17218393211855787),
            ("IIIZ", +0.17218393211855818),
            ("IZIZ", +0.16614543242281926),
            ("ZZIZ", +0.16614543242281926),
            ("IIZZ", -0.22575349071287362),
            ("IZZZ", +0.16892753854646372),
            ("ZZZZ", +0.17464343053355980),
            ("IXIX", +0.04523279999117751),
            ("ZXIX", +0.04523279999117751),
            ("IXZX", -0.04523279999117751),
            ("ZXZX", -0.04523279999117751),
        ]
    )

    REF_H2_PARITY_2Q_REDUCED = SparsePauliOp.from_list(
        [
            ("II", -1.05237324646359750),
            ("IZ", +0.39793742283143163),
            ("ZI", -0.39793742283143140),
            ("ZZ", -0.01128010423438501),
            ("XX", +0.18093119996471000),
        ]
    )

    REF_H2_PARITY_TAPERED = SparsePauliOp.from_list(
        [("I", -1.04109314), ("Z", -0.79587485), ("X", -0.1809312)]
    )

    def setUp(self):
        super().setUp()
        driver = PySCFDriver()
        self.driver_result = driver.run()
        self.num_particles = self.driver_result.num_particles
        self.h2_op, _ = self.driver_result.second_q_ops()

    def test_z2_symmetry(self):
        """Test mapping to qubit operator with z2 symmetry tapering"""
        mapper = JordanWignerMapper()

        with self.subTest("Previous"):
            sector_locator = self.driver_result.symmetry_sector_locator
            qubit_conv = QubitConverter(mapper, z2symmetry_reduction="auto")
            qubit_op = qubit_conv.convert(self.h2_op, sector_locator=sector_locator).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED)

        with self.subTest("After"):
            tapered_qubit_mapper = TaperedQubitMapper.from_problem(mapper, self.driver_result)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED)

        with self.subTest("From Z2Symmetry object"):
            z2_sym = Z2Symmetries(
                symmetries=PauliList(["ZIIZ", "ZIZI", "ZZII"]),
                sq_paulis=PauliList(["IIIX", "IIXI", "IXII"]),
                sq_list=[0, 1, 2],
                tapering_values=[-1, 1, -1],
            )
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED)

        with self.subTest("From empty Z2Symmetry object"):
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW))

        with self.subTest("From Z2Symmetry object no tapering values"):
            z2_sym = Z2Symmetries(
                symmetries=PauliList(["ZIIZ", "ZIZI", "ZZII"]),
                sq_paulis=PauliList(["IIIX", "IIXI", "IXII"]),
                sq_list=[0, 1, 2],
            )
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = [op.primitive for op in tapered_qubit_mapper.map(self.h2_op)]
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED_LIST)

        with self.subTest("From Z2Symmetry object automatic"):
            qubit_op = mapper.map(self.h2_op).primitive
            z2_sym = Z2Symmetries.find_z2_symmetries(qubit_op)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2_sym)
            qubit_op = [op.primitive for op in tapered_qubit_mapper.map(self.h2_op)]
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED_LIST)

    def test_z2_symmetry_two_qubit_reduction(self):
        """Test mapping to qubit operator with z2 symmetry tapering and two qubit reduction"""

        with self.subTest("Two qubit reduction set to False and No particle number"):
            mapper = ParityMapper(num_particles=None)
            tapered_qubit_mapper = TaperedQubitMapper.from_problem(mapper, self.driver_result)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_PARITY_TAPERED)

        with self.subTest("Two qubit reduction set to False and particle number (1, 1)"):
            mapper = ParityMapper(num_particles=(1, 1))
            tapered_qubit_mapper = TaperedQubitMapper.from_problem(mapper, self.driver_result)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_PARITY_TAPERED)

        with self.subTest("Two qubit reduction set to True and No particle number"):
            mapper = ParityMapper(num_particles=None)
            tapered_qubit_mapper = TaperedQubitMapper.from_problem(mapper, self.driver_result)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_PARITY_TAPERED)

        with self.subTest("Two qubit reduction set to True and particle number (1, 1)"):
            mapper = ParityMapper(num_particles=(1, 1))
            tapered_qubit_mapper = TaperedQubitMapper.from_problem(mapper, self.driver_result)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_PARITY_TAPERED)

    def test_empty_z2_symmetry_two_qubit_reduction(self):
        """Test mapping to qubit operator with empty z2 symmetry tapering and two qubit reduction"""

        with self.subTest("Two qubit reduction set to False and No particle number"):
            mapper = ParityMapper(num_particles=None)
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_PARITY))

        with self.subTest("Two qubit reduction set to False and particle number (1, 1)"):
            mapper = ParityMapper(num_particles=(1, 1))
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_PARITY))

        with self.subTest("Two qubit reduction set to True and No particle number"):
            mapper = ParityMapper(num_particles=None)
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_PARITY))

        with self.subTest("Two qubit reduction set to True and particle number (1, 1)"):
            mapper = ParityMapper(num_particles=(1, 1))
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_PARITY_2Q_REDUCED)

    def test_taperedqubitmapper_jw_mapper(self):
        """Test Tapered Qubit Mapper with Jordan Wigner Mapper"""

        with self.subTest("Tapered Qubit Mapper from problem"):
            mapper = JordanWignerMapper()
            tapered_qubit_mapper = TaperedQubitMapper.from_problem(mapper, self.driver_result)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED))

        with self.subTest("Tapered Qubit Mapper from empty symmetry"):
            mapper = JordanWignerMapper()
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op).primitive
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW))


if __name__ == "__main__":
    unittest.main()
