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

import numpy as np

from qiskit_nature.settings import settings
from qiskit_nature.circuit.library.ansatzes.uccsd import UCCSD
from qiskit_nature.initializers import MP2Initializer
from test import QiskitNatureTestCase

from ddt import data, ddt, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.circuit.library import UCC
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import FermionicOp


def assert_ucc_like_ansatz(test_case, ansatz, num_spin_orbitals, expected_ops):
    """Assertion utility."""
    excitation_ops = ansatz.excitation_ops()

    test_case.assertEqual(len(excitation_ops), len(expected_ops))
    for op, exp in zip(excitation_ops, expected_ops):
        test_case.assertListEqual(op.to_list(), exp.to_list())

    ansatz._build()
    test_case.assertEqual(ansatz.num_qubits, num_spin_orbitals)


@ddt
class TestUCC(QiskitNatureTestCase):
    """Tests for the UCC Ansatz."""

    # Note: many variations of this class are tested by its sub-classes UCCSD, PUCCD and SUCCD.
    # Thus, the tests here mainly cover edge cases which those classes cannot account for.

    @unpack
    @data(
        (
            "t",
            8,
            (2, 2),
            [
                FermionicOp([("++--+I-I", 1j), ("--++-I+I", 1j)], display_format="dense"),
                FermionicOp([("++--+II-", 1j), ("--++-II+", 1j)], display_format="dense"),
                FermionicOp([("++--I+-I", 1j), ("--++I-+I", 1j)], display_format="dense"),
                FermionicOp([("++--I+I-", 1j), ("--++I-I+", 1j)], display_format="dense"),
                FermionicOp([("+I-I++--", 1j), ("-I+I--++", 1j)], display_format="dense"),
                FermionicOp([("+II-++--", 1j), ("-II+--++", 1j)], display_format="dense"),
                FermionicOp([("I+-I++--", 1j), ("I-+I--++", 1j)], display_format="dense"),
                FermionicOp([("I+I-++--", 1j), ("I-I+--++", 1j)], display_format="dense"),
            ],
        ),
        (
            "t",
            8,
            (2, 1),
            [
                FermionicOp([("++--+-II", 1j), ("--++-+II", 1j)], display_format="dense"),
                FermionicOp([("++--+I-I", 1j), ("--++-I+I", 1j)], display_format="dense"),
                FermionicOp([("++--+II-", 1j), ("--++-II+", 1j)], display_format="dense"),
            ],
        ),
        (
            "q",
            8,
            (2, 2),
            [FermionicOp([("++--++--", 1j), ("--++--++", -1j)], display_format="dense")],
        ),
        # TODO: add more edge cases?
    )
    def test_ucc_ansatz(self, excitations, num_spin_orbitals, num_particles, expected):
        """Tests the UCC Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCC(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            excitations=excitations,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expected)

    @unpack
    @data(
        # Excitations not a list of pairs
        (
            8,
            (2, 2),
            [((0, 1, 4), (2, 3, 6), (2, 3, 7))],
        ),
        # Excitation pair has not same length
        (
            8,
            (2, 2),
            [((0, 1, 4), (2, 3, 6, 7))],
        ),
        # Excitation pair with non-unique indices
        (
            8,
            (2, 2),
            [((0, 1, 4), (2, 4, 6))],
        ),
        (
            8,
            (2, 2),
            [((0, 1, 1), (2, 3, 6))],
        ),
    )
    def test_custom_excitations(self, num_spin_orbitals, num_particles, excitations):
        """Tests if an error is raised when the excitations have a wrong format"""
        converter = QubitConverter(JordanWignerMapper())

        # pylint: disable=unused-argument
        def custom_excitations(num_spin_orbitals, num_particles):
            return excitations

        ansatz = UCC(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            excitations=custom_excitations,
        )

        with self.assertRaises(QiskitNatureError):
            ansatz.excitation_ops()

    @unpack
    @data(
        # No initializer
        (8, (2, 2), None, np.zeros(26)),
    )
    def test_with_no_initializer(self, num_spin_orbitals, num_particles, initializer, expected):
        """Test with no initializer."""

        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCCSD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            initializer=initializer,
        )
        ansatz._build()

        with self.subTest("test initial_point"):
            np.testing.assert_array_equal(ansatz.initial_point, expected)

        with self.subTest("test preferred_init_points"):
            np.testing.assert_array_equal(ansatz.preferred_init_points, expected)

    @unpack
    @data(
        # MP2Initializer h2 molecule
        (
            4,
            (1, 1),
            np.asarray([-0.595463473004833, 0.714165284454473]),
            np.asarray(
                [
                    [
                        [[0.6823895331520421, 0.0], [0.0, 0.6707327783087585]],
                        [[0.0, 0.17900057606140662], [0.17900057606140676, 0.0]],
                    ],
                    [
                        [[0.0, 0.17900057606140674], [0.17900057606140682, 0.0]],
                        [[0.6707327783087589, 0.0], [0.0, 0.7051056321727855]],
                    ],
                ],
            ),
            np.asarray([0.0, 0.0, -0.06834019757197064]),
        )
    )
    def test_with_mp2_initializer(
        self, num_spin_orbitals, num_particles, orbital_energies, integral_matrix, expected
    ):
        """Test with MP2 initializer."""

        initializer = MP2Initializer(num_spin_orbitals, orbital_energies, integral_matrix)

        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCCSD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            initializer=initializer,
        )
        ansatz._build()

        with self.subTest("test initial_point"):
            np.testing.assert_array_almost_equal(ansatz.initial_point, expected)

        with self.subTest("test preferred_init_points"):
            np.testing.assert_array_almost_equal(ansatz.preferred_init_points, expected)
