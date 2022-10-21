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

"""Test the UCC Ansatz."""

from test import QiskitNatureTestCase

import unittest

from ddt import data, ddt, unpack

from qiskit import transpile
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators import FermionicOp


def assert_ucc_like_ansatz(test_case, ansatz, num_spatial_orbitals, expected_ops):
    """Assertion utility."""
    excitation_ops = ansatz.excitation_ops()

    test_case.assertEqual(len(excitation_ops), len(expected_ops))
    for op, exp in zip(excitation_ops, expected_ops):
        test_case.assertEqual(op, exp)

    ansatz._build()
    test_case.assertEqual(ansatz.num_qubits, 2 * num_spatial_orbitals)


@ddt
class TestUCC(QiskitNatureTestCase):
    """Tests for the UCC Ansatz."""

    # Note: many variations of this class are tested by its sub-classes UCCSD, PUCCD and SUCCD.
    # Thus, the tests here mainly cover edge cases which those classes cannot account for.

    @unpack
    @data(
        (
            "t",
            4,
            (2, 2),
            [
                FermionicOp(
                    {"+_0 +_1 +_4 -_2 -_3 -_6": 1j, "+_6 +_3 +_2 -_4 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_1 +_4 -_2 -_3 -_7": 1j, "+_7 +_3 +_2 -_4 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_1 +_5 -_2 -_3 -_6": 1j, "+_6 +_3 +_2 -_5 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_1 +_5 -_2 -_3 -_7": 1j, "+_7 +_3 +_2 -_5 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_4 +_5 -_2 -_6 -_7": 1j, "+_7 +_6 +_2 -_5 -_4 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_4 +_5 -_3 -_6 -_7": 1j, "+_7 +_6 +_3 -_5 -_4 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_1 +_4 +_5 -_2 -_6 -_7": 1j, "+_7 +_6 +_2 -_5 -_4 -_1": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_1 +_4 +_5 -_3 -_6 -_7": 1j, "+_7 +_6 +_3 -_5 -_4 -_1": -1j},
                    num_spin_orbitals=8,
                ),
            ],
        ),
        (
            "t",
            4,
            (2, 1),
            [
                FermionicOp(
                    {"+_0 +_1 +_4 -_2 -_3 -_5": 1j, "+_5 +_3 +_2 -_4 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_1 +_4 -_2 -_3 -_6": 1j, "+_6 +_3 +_2 -_4 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
                FermionicOp(
                    {"+_0 +_1 +_4 -_2 -_3 -_7": 1j, "+_7 +_3 +_2 -_4 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                ),
            ],
        ),
        (
            "q",
            4,
            (2, 2),
            [
                FermionicOp(
                    {"+_0 +_1 +_4 +_5 -_2 -_3 -_6 -_7": 1j, "+_7 +_6 +_3 +_2 -_5 -_4 -_1 -_0": -1j},
                    num_spin_orbitals=8,
                )
            ],
        ),
        # TODO: add more edge cases?
    )
    def test_ucc_ansatz(self, excitations, num_spatial_orbitals, num_particles, expect):
        """Tests the UCC Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCC(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spatial_orbitals=num_spatial_orbitals,
            excitations=excitations,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spatial_orbitals, expect)

    @unpack
    @data(
        # Excitations not a list of pairs
        (
            4,
            (2, 2),
            [((0, 1, 4), (2, 3, 6), (2, 3, 7))],
        ),
        # Excitation pair has not same length
        (
            4,
            (2, 2),
            [((0, 1, 4), (2, 3, 6, 7))],
        ),
        # Excitation pair with non-unique indices
        (
            4,
            (2, 2),
            [((0, 1, 4), (2, 4, 6))],
        ),
        (
            4,
            (2, 2),
            [((0, 1, 1), (2, 3, 6))],
        ),
    )
    def test_custom_excitations(self, num_spatial_orbitals, num_particles, excitations):
        """Tests if an error is raised when the excitations have a wrong format"""
        converter = QubitConverter(JordanWignerMapper())

        # pylint: disable=unused-argument
        def custom_excitations(num_spatial_orbitals, num_particles):
            return excitations

        with self.assertRaises(QiskitNatureError):
            ansatz = UCC(
                qubit_converter=converter,
                num_particles=num_particles,
                num_spatial_orbitals=num_spatial_orbitals,
                excitations=custom_excitations,
            )
            ansatz.excitation_ops()

    def test_transpile_no_parameters(self):
        """Test transpilation without parameters"""

        num_spatial_orbitals = 4
        num_particles = (2, 2)
        qubit_converter = QubitConverter(mapper=JordanWignerMapper())

        ansatz = UCC(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_converter=qubit_converter,
            excitations="s",
        )

        ansatz = transpile(ansatz, optimization_level=3)
        self.assertEqual(ansatz.num_qubits, 8)

    def test_build_ucc(self):
        """Test building UCC"""
        ucc = UCC()

        with self.subTest("Check defaulted construction"):
            self.assertIsNone(ucc.num_particles)
            self.assertIsNone(ucc.num_spatial_orbitals)
            self.assertIsNone(ucc.excitations)
            self.assertIsNone(ucc.qubit_converter)
            self.assertIsNone(ucc.operators)
            self.assertIsNone(ucc.excitation_list)
            self.assertEqual(ucc.num_qubits, 0)
            with self.assertRaises(ValueError):
                _ = ucc.data

        with self.subTest("Set num particles"):
            ucc.num_particles = (1, 1)
            self.assertEqual(ucc.num_particles, (1, 1))
            self.assertIsNone(ucc.operators)
            with self.assertRaises(ValueError):
                _ = ucc.data

        with self.subTest("Set num spatial orbitals"):
            ucc.num_spatial_orbitals = 2
            self.assertEqual(ucc.num_spatial_orbitals, 2)
            self.assertIsNone(ucc.operators)
            with self.assertRaises(ValueError):
                _ = ucc.data

        with self.subTest("Set excitations"):
            ucc.excitations = "sd"
            self.assertEqual(ucc.excitations, "sd")
            self.assertIsNone(ucc.operators)
            with self.assertRaises(ValueError):
                _ = ucc.data

        with self.subTest("Set qubit converter to complete build"):
            converter = QubitConverter(JordanWignerMapper())
            ucc.qubit_converter = converter
            self.assertEqual(ucc.qubit_converter, converter)
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(len(ucc.operators), 3)
            self.assertEqual(ucc.num_qubits, 4)
            self.assertIsNotNone(ucc.data)

        with self.subTest("Set custom operators"):
            self.assertEqual(len(ucc.operators), 3)
            ucc.operators = ucc.operators[:2]
            self.assertEqual(len(ucc.operators), 2)
            self.assertEqual(ucc.num_qubits, 4)

        with self.subTest("Reset operators back to as per UCC"):
            ucc.operators = None
            self.assertEqual(ucc.num_qubits, 4)
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(len(ucc.operators), 3)

        with self.subTest("Set num particles to include 0"):
            ucc.num_particles = (1, 0)
            self.assertEqual(ucc.num_particles, (1, 0))
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(len(ucc.operators), 1)

        with self.subTest("Change num particles"):
            ucc.num_particles = (1, 1)
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(len(ucc.operators), 3)

        with self.subTest("Change num spatial orbitals"):
            ucc.num_spatial_orbitals = 3
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(len(ucc.operators), 8)

        with self.subTest("Change excitations"):
            ucc.excitations = "s"
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(len(ucc.operators), 4)

        with self.subTest("Change qubit converter"):
            ucc.qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
            # Has not been used to convert so we need to force it to do two qubit reduction
            ucc.qubit_converter.force_match(num_particles=ucc.num_particles)
            self.assertIsNotNone(ucc.operators)
            self.assertEqual(ucc.num_qubits, 4)


if __name__ == "__main__":
    unittest.main()
