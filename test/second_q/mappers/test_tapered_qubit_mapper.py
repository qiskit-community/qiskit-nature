# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TaperedQubitMapper."""

import unittest
from test import QiskitNatureTestCase

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
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

    REF_H2_JW_CLIF = SparsePauliOp.from_list(
        [
            ("IIII", -0.8105479805373267),
            ("ZIIX", +0.1721839326191554),
            ("ZIXI", -0.2257534922240235),
            ("ZXII", +0.17218393261915513),
            ("ZIII", -0.22575349222402355),
            ("IIXX", +0.12091263261776627),
            ("IXIX", +0.16892753870087898),
            ("XIII", +0.0452327999460578),
            ("XXII", -0.0452327999460578),
            ("XIXX", -0.0452327999460578),
            ("XXXX", +0.0452327999460578),
            ("IIIX", +0.16614543256382402),
            ("IXXI", +0.166145432563824),
            ("IIXI", +0.17464343068300442),
            ("IXII", +0.12091263261776627),
        ]
    )

    REF_H2_JW_TAPERED = SparsePauliOp.from_list(
        [("I", -1.04109314), ("Z", -0.79587485), ("X", 0.1809312)]
    )

    REF_H2_PT = SparsePauliOp.from_list(
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

    REF_H2_PT_CLIF = SparsePauliOp.from_list(
        [
            ("IIII", -0.8105479805373267),
            ("IZIX", +0.1721839326191554),
            ("IZXX", -0.2257534922240235),
            ("IZXI", +0.17218393261915513),
            ("XZII", -0.22575349222402355),
            ("IIXI", +0.12091263261776627),
            ("IIXX", +0.16892753870087898),
            ("XXIX", +0.0452327999460578),
            ("IXXX", -0.0452327999460578),
            ("XXXX", -0.0452327999460578),
            ("IXIX", +0.0452327999460578),
            ("XIIX", +0.16614543256382402),
            ("IIIX", +0.166145432563824),
            ("XIXX", +0.17464343068300442),
            ("XIXI", +0.12091263261776627),
        ]
    )

    REF_H2_PT_2Q_REDUCED = SparsePauliOp.from_list(
        [
            ("II", -1.05237324646359750),
            ("IZ", +0.39793742283143163),
            ("ZI", -0.39793742283143140),
            ("ZZ", -0.01128010423438501),
            ("XX", +0.18093119996471000),
        ]
    )

    REF_H2_PT_TAPERED = SparsePauliOp.from_list(
        [("I", -1.04109314), ("Z", -0.79587485), ("X", -0.1809312)]
    )

    def setUp(self):
        super().setUp()
        driver = PySCFDriver()
        self.driver_result = driver.run()
        self.num_particles = self.driver_result.num_particles
        self.h2_op, _ = self.driver_result.second_q_ops()
        self.jw_mapper = JordanWignerMapper()
        self.pt_mapper = ParityMapper()

    def test_z2_symmetry(self):
        """Test mapping to qubit operator with z2 symmetry tapering"""
        mapper = JordanWignerMapper()

        with self.subTest("TaperedQubitMapper"):
            tapered_qubit_mapper = self.driver_result.get_tapered_mapper(mapper)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED)

        with self.subTest("From Z2Symmetry object"):
            z2_sym = Z2Symmetries(
                symmetries=[Pauli("ZIIZ"), Pauli("ZIZI"), Pauli("ZZII")],
                sq_paulis=[Pauli("IIIX"), Pauli("IIXI"), Pauli("IXII")],
                sq_list=[0, 1, 2],
                tapering_values=[-1, 1, -1],
            )
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_JW_TAPERED)

        with self.subTest("From empty Z2Symmetry object"):
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW))

        with self.subTest("From Z2Symmetry object no tapering values"):
            z2_sym = Z2Symmetries(
                symmetries=[Pauli("ZIIZ"), Pauli("ZIZI"), Pauli("ZZII")],
                sq_paulis=[Pauli("IIIX"), Pauli("IIXI"), Pauli("IXII")],
                sq_list=[0, 1, 2],
            )
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))
            tapered_qubit_mapper.z2symmetries.tapering_values = [-1, 1, -1]
            qubit_op = tapered_qubit_mapper.taper_clifford(qubit_op)
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED))

        with self.subTest("From Z2Symmetry object automatic but no sector locator"):
            qubit_op = mapper.map(self.h2_op)
            z2_sym = Z2Symmetries.find_z2_symmetries(qubit_op)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))

    def test_z2_symmetry_two_qubit_reduction(self):
        """Test mapping to qubit operator with z2 symmetry tapering and two qubit reduction"""
        with self.subTest("No 2-qubit reduction in the ParityMapper"):
            mapper = ParityMapper(num_particles=None)
            tapered_qubit_mapper = self.driver_result.get_tapered_mapper(mapper)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertEqual(qubit_op, TestTaperedQubitMapper.REF_H2_PT_TAPERED)

    def test_empty_z2_symmetry_two_qubit_reduction(self):
        """Test mapping to qubit operator with empty z2 symmetry tapering and two qubit reduction"""
        with self.subTest("No 2-qubit reduction in the ParityMapper"):
            mapper = ParityMapper(num_particles=None)
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_PT))

        with self.subTest("With 2-qubit reduction in the ParityMapper"):
            mapper = ParityMapper(num_particles=(1, 1))
            z2_sym = Z2Symmetries([], [], [], None)
            tapered_qubit_mapper = TaperedQubitMapper(mapper, z2symmetries=z2_sym)
            qubit_op = tapered_qubit_mapper.map(self.h2_op)
            self.assertTrue(qubit_op.equiv(TestTaperedQubitMapper.REF_H2_PT_2Q_REDUCED))

    def test_map_clifford(self):
        """Test the first exposed step of the mapping. Mapping to Pauli operators and composing with
        symmetry cliffords"""
        with self.subTest("Single operator JW"):
            jw_tqm = TaperedQubitMapper(self.jw_mapper)
            jw_op_h2 = jw_tqm.map_clifford(self.h2_op)
            self.assertTrue(jw_op_h2.equiv(TestTaperedQubitMapper.REF_H2_JW))

            jw_tqm = self.driver_result.get_tapered_mapper(self.jw_mapper)
            jw_op_h2 = jw_tqm.map_clifford(self.h2_op)
            self.assertTrue(jw_op_h2.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))
            # Compose with symmetry cliffords if z2 not empty even if Num_particles is empty
            jw_tqm.z2symmetries.tapering_values = None
            jw_op_h2 = jw_tqm.map_clifford(self.h2_op)
            self.assertTrue(jw_op_h2.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))

        with self.subTest("Single operator PT"):
            pt_tqm = TaperedQubitMapper(self.pt_mapper)
            pt_op_h2 = pt_tqm.map_clifford(self.h2_op)
            self.assertTrue(pt_op_h2.equiv(TestTaperedQubitMapper.REF_H2_PT))

            pt_tqm = self.driver_result.get_tapered_mapper(self.pt_mapper)
            pt_op_h2 = pt_tqm.map_clifford(self.h2_op)
            self.assertTrue(pt_op_h2.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))
            # Compose with symmetry cliffords if z2 not empty even with empty tapering values
            pt_tqm.z2symmetries.tapering_values = None
            pt_op_h2 = pt_tqm.map_clifford(self.h2_op)
            self.assertTrue(pt_op_h2.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))

        with self.subTest("Dictionary / List of operators and JW Mapper"):
            jw_tqm = TaperedQubitMapper(self.jw_mapper)
            jw_op_h2_list = jw_tqm.map_clifford([self.h2_op])
            jw_op_h2_dict = jw_tqm.map_clifford({"h2": self.h2_op})
            self.assertTrue(isinstance(jw_op_h2_list, list))
            self.assertTrue(isinstance(jw_op_h2_dict, dict))
            self.assertEqual(len(jw_op_h2_list), 1)
            self.assertEqual(len(jw_op_h2_dict), 1)
            sparse_jw_op_h2_list = jw_op_h2_list[0]
            sparse_jw_op_h2_dict = jw_op_h2_dict["h2"]
            self.assertTrue(sparse_jw_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_JW))
            self.assertTrue(sparse_jw_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_JW))

            jw_tqm = self.driver_result.get_tapered_mapper(self.jw_mapper)
            jw_op_h2_list = jw_tqm.map_clifford([self.h2_op])
            jw_op_h2_dict = jw_tqm.map_clifford({"h2": self.h2_op})
            self.assertTrue(isinstance(jw_op_h2_list, list))
            self.assertTrue(isinstance(jw_op_h2_dict, dict))
            self.assertEqual(len(jw_op_h2_list), 1)
            self.assertEqual(len(jw_op_h2_dict), 1)
            sparse_jw_op_h2_list = jw_op_h2_list[0]
            sparse_jw_op_h2_dict = jw_op_h2_dict["h2"]
            self.assertTrue(sparse_jw_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))
            self.assertTrue(sparse_jw_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))

        with self.subTest("Dictionary / List of operators and PT Mapper"):
            pt_tqm = TaperedQubitMapper(self.pt_mapper)
            pt_op_h2_list = pt_tqm.map_clifford([self.h2_op])
            pt_op_h2_dict = pt_tqm.map_clifford({"h2": self.h2_op})
            self.assertTrue(isinstance(pt_op_h2_list, list))
            self.assertTrue(isinstance(pt_op_h2_dict, dict))
            self.assertEqual(len(pt_op_h2_list), 1)
            self.assertEqual(len(pt_op_h2_dict), 1)
            sparse_pt_op_h2_list = pt_op_h2_list[0]
            sparse_pt_op_h2_dict = pt_op_h2_dict["h2"]
            self.assertTrue(sparse_pt_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_PT))
            self.assertTrue(sparse_pt_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_PT))

            pt_tqm = self.driver_result.get_tapered_mapper(self.pt_mapper)
            pt_op_h2_list = pt_tqm.map_clifford([self.h2_op])
            pt_op_h2_dict = pt_tqm.map_clifford({"h2": self.h2_op})
            self.assertTrue(isinstance(pt_op_h2_list, list))
            self.assertTrue(isinstance(pt_op_h2_dict, dict))
            self.assertEqual(len(pt_op_h2_list), 1)
            self.assertEqual(len(pt_op_h2_dict), 1)
            sparse_pt_op_h2_list = pt_op_h2_list[0]
            sparse_pt_op_h2_dict = pt_op_h2_dict["h2"]
            self.assertTrue(sparse_pt_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))
            self.assertTrue(sparse_pt_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))

    def test_taper_clifford(self):
        """Test the second exposed step of the mapping. Applying the symmetry reduction"""
        with self.subTest("Single operator JW"):
            jw_tqm = TaperedQubitMapper(self.jw_mapper)
            jw_op_h2 = jw_tqm.map_clifford(self.h2_op)
            jw_op_h2_tap = jw_tqm.taper_clifford(jw_op_h2)
            self.assertTrue(jw_op_h2.equiv(TestTaperedQubitMapper.REF_H2_JW))
            self.assertTrue(jw_op_h2_tap.equiv(TestTaperedQubitMapper.REF_H2_JW))

            jw_tqm = self.driver_result.get_tapered_mapper(self.jw_mapper)
            jw_op_h2 = jw_tqm.map_clifford(self.h2_op)
            jw_op_h2_tap = jw_tqm.taper_clifford(jw_op_h2)
            self.assertTrue(jw_op_h2.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))
            self.assertTrue(jw_op_h2_tap.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED))

        with self.subTest("Single operator PT"):
            pt_tqm = TaperedQubitMapper(self.pt_mapper)
            pt_op_h2 = pt_tqm.map_clifford(self.h2_op)
            pt_op_h2_tap = pt_tqm.taper_clifford(pt_op_h2)
            self.assertTrue(pt_op_h2.equiv(TestTaperedQubitMapper.REF_H2_PT))
            self.assertTrue(pt_op_h2_tap.equiv(TestTaperedQubitMapper.REF_H2_PT))

            pt_tqm = self.driver_result.get_tapered_mapper(self.pt_mapper)
            pt_op_h2 = pt_tqm.map_clifford(self.h2_op)
            pt_op_h2_tap = pt_tqm.taper_clifford(pt_op_h2)
            self.assertTrue(pt_op_h2.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))
            self.assertTrue(pt_op_h2_tap.equiv(TestTaperedQubitMapper.REF_H2_PT_TAPERED))

        with self.subTest("Dictionary / List of operators and JW Mapper"):
            # Passive TaperedQubitMapper when not associated with a symmetry
            jw_tqm = TaperedQubitMapper(self.jw_mapper)
            jw_op_h2_list = jw_tqm.map_clifford([self.h2_op])
            jw_op_h2_tap_list = jw_tqm.taper_clifford(jw_op_h2_list)
            jw_op_h2_dict = jw_tqm.map_clifford({"h2": self.h2_op})
            jw_op_h2_tap_dict = jw_tqm.taper_clifford(jw_op_h2_dict)
            self.assertTrue(isinstance(jw_op_h2_list, list))
            self.assertTrue(isinstance(jw_op_h2_tap_list, list))
            self.assertTrue(isinstance(jw_op_h2_tap_dict, dict))
            self.assertEqual(len(jw_op_h2_list), 1)
            self.assertEqual(len(jw_op_h2_tap_list), 1)
            self.assertEqual(len(jw_op_h2_tap_dict), 1)
            sparse_jw_op_h2_list = jw_op_h2_list[0]
            sparse_jw_op_h2_tap_list = jw_op_h2_tap_list[0]
            sparse_jw_op_h2_dict = jw_op_h2_dict["h2"]
            sparse_jw_op_h2_tap_dict = jw_op_h2_tap_dict["h2"]
            self.assertTrue(sparse_jw_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_JW))
            self.assertTrue(sparse_jw_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_JW))
            self.assertTrue(sparse_jw_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_JW))
            self.assertTrue(sparse_jw_op_h2_tap_dict.equiv(TestTaperedQubitMapper.REF_H2_JW))

            # TaperedQubitMapper created from the problem
            jw_tqm = self.driver_result.get_tapered_mapper(self.jw_mapper)
            jw_op_h2_list = jw_tqm.map_clifford([self.h2_op])
            jw_op_h2_tap_list = jw_tqm.taper_clifford(jw_op_h2_list)
            jw_op_h2_dict = jw_tqm.map_clifford({"h2": self.h2_op})
            jw_op_h2_tap_dict = jw_tqm.taper_clifford(jw_op_h2_dict)
            self.assertTrue(isinstance(jw_op_h2_list, list))
            self.assertTrue(isinstance(jw_op_h2_tap_list, list))
            self.assertTrue(isinstance(jw_op_h2_tap_dict, dict))
            self.assertEqual(len(jw_op_h2_list), 1)
            self.assertEqual(len(jw_op_h2_tap_list), 1)
            self.assertEqual(len(jw_op_h2_tap_dict), 1)
            sparse_jw_op_h2_list = jw_op_h2_list[0]
            sparse_jw_op_h2_tap_list = jw_op_h2_tap_list[0]
            sparse_jw_op_h2_dict = jw_op_h2_dict["h2"]
            sparse_jw_op_h2_tap_dict = jw_op_h2_tap_dict["h2"]
            self.assertTrue(sparse_jw_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))
            self.assertTrue(
                sparse_jw_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED)
            )
            self.assertTrue(sparse_jw_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_JW_CLIF))
            self.assertTrue(
                sparse_jw_op_h2_tap_dict.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED)
            )

        with self.subTest("Dictionary / List of operators and PT Mapper"):
            # Passive TaperedQubitMapper when not associated with a symmetry
            pt_tqm = TaperedQubitMapper(self.pt_mapper)
            pt_op_h2_list = pt_tqm.map_clifford([self.h2_op])
            pt_op_h2_tap_list = pt_tqm.taper_clifford(pt_op_h2_list)
            pt_op_h2_dict = pt_tqm.map_clifford({"h2": self.h2_op})
            pt_op_h2_tap_dict = pt_tqm.taper_clifford(pt_op_h2_dict)
            self.assertTrue(isinstance(pt_op_h2_list, list))
            self.assertTrue(isinstance(pt_op_h2_tap_list, list))
            self.assertTrue(isinstance(pt_op_h2_tap_dict, dict))
            self.assertEqual(len(pt_op_h2_list), 1)
            self.assertEqual(len(pt_op_h2_tap_list), 1)
            self.assertEqual(len(pt_op_h2_tap_dict), 1)
            sparse_pt_op_h2_list = pt_op_h2_list[0]
            sparse_pt_op_h2_tap_list = pt_op_h2_tap_list[0]
            sparse_pt_op_h2_dict = pt_op_h2_dict["h2"]
            sparse_pt_op_h2_tap_dict = pt_op_h2_tap_dict["h2"]
            self.assertTrue(sparse_pt_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_PT))
            self.assertTrue(sparse_pt_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_PT))
            self.assertTrue(sparse_pt_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_PT))
            self.assertTrue(sparse_pt_op_h2_tap_dict.equiv(TestTaperedQubitMapper.REF_H2_PT))

            # TaperedQubitMapper created from problem
            pt_tqm = self.driver_result.get_tapered_mapper(self.pt_mapper)
            pt_op_h2_list = pt_tqm.map_clifford([self.h2_op])
            pt_op_h2_tap_list = pt_tqm.taper_clifford(pt_op_h2_list)
            pt_op_h2_dict = pt_tqm.map_clifford({"h2": self.h2_op})
            pt_op_h2_tap_dict = pt_tqm.taper_clifford(pt_op_h2_dict)
            self.assertTrue(isinstance(pt_op_h2_list, list))
            self.assertTrue(isinstance(pt_op_h2_tap_list, list))
            self.assertTrue(isinstance(pt_op_h2_tap_dict, dict))
            self.assertEqual(len(pt_op_h2_list), 1)
            self.assertEqual(len(pt_op_h2_tap_list), 1)
            self.assertEqual(len(pt_op_h2_tap_dict), 1)
            sparse_pt_op_h2_list = pt_op_h2_list[0]
            sparse_pt_op_h2_tap_list = pt_op_h2_tap_list[0]
            sparse_pt_op_h2_dict = pt_op_h2_dict["h2"]
            sparse_pt_op_h2_tap_dict = pt_op_h2_tap_dict["h2"]
            self.assertTrue(sparse_pt_op_h2_list.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))
            self.assertTrue(
                sparse_pt_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_PT_TAPERED)
            )
            self.assertTrue(sparse_pt_op_h2_dict.equiv(TestTaperedQubitMapper.REF_H2_PT_CLIF))
            self.assertTrue(
                sparse_pt_op_h2_tap_dict.equiv(TestTaperedQubitMapper.REF_H2_PT_TAPERED)
            )

        with self.subTest("Check Commutes"):
            jw_tqm = self.driver_result.get_tapered_mapper(self.jw_mapper)

            ops = [
                TestTaperedQubitMapper.REF_H2_JW_CLIF,
                SparsePauliOp.from_list([("IXYZ", 1.0)]),
            ]
            jw_op_h2_tap_list = jw_tqm.taper_clifford(ops, suppress_none=False)
            self.assertTrue(isinstance(jw_op_h2_tap_list, list))
            self.assertEqual(len(jw_op_h2_tap_list), 2)
            sparse_jw_op_h2_tap_list = jw_op_h2_tap_list[0]
            self.assertTrue(
                sparse_jw_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED)
            )
            self.assertTrue(jw_op_h2_tap_list[1] is None)

            # suppress none to True should affect the second operator which does not commute with the
            # symmetry.
            jw_op_h2_tap_list = jw_tqm.taper_clifford(ops, suppress_none=True)
            self.assertTrue(isinstance(jw_op_h2_tap_list, list))
            self.assertEqual(len(jw_op_h2_tap_list), 1)
            sparse_jw_op_h2_tap_list = jw_op_h2_tap_list[0]
            self.assertTrue(
                sparse_jw_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED)
            )

            jw_op_h2_tap_list = jw_tqm.taper_clifford(ops, check_commutes=False, suppress_none=True)
            self.assertTrue(isinstance(jw_op_h2_tap_list, list))
            self.assertEqual(len(jw_op_h2_tap_list), 2)
            sparse_jw_op_h2_tap_list = jw_op_h2_tap_list[0]
            self.assertTrue(
                sparse_jw_op_h2_tap_list.equiv(TestTaperedQubitMapper.REF_H2_JW_TAPERED)
            )
            self.assertTrue(jw_op_h2_tap_list[1] == SparsePauliOp.from_list([("I", 1.0)]))


if __name__ == "__main__":
    unittest.main()
