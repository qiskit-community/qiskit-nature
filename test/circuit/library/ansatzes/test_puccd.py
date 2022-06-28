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

"""Test the PUCCD Ansatz."""

from test import QiskitNatureTestCase
from test.circuit.library.ansatzes.test_ucc import assert_ucc_like_ansatz

from ddt import ddt, data, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.circuit.library import PUCCD
from qiskit_nature.second_quantization.operators.fermionic import JordanWignerMapper
from qiskit_nature.second_quantization.operators import FermionicOp
from qiskit_nature.second_quantization.operators import QubitConverter


@ddt
class TestPUCC(QiskitNatureTestCase):
    """Tests for the PUCCD Ansatz."""

    @unpack
    @data(
        (4, (1, 1), [FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense")]),
        (
            8,
            (2, 2),
            [
                FermionicOp([("+I-I+I-I", 1j), ("-I+I-I+I", -1j)], display_format="dense"),
                FermionicOp([("+II-+II-", 1j), ("-II+-II+", -1j)], display_format="dense"),
                FermionicOp([("I+-II+-I", 1j), ("I-+II-+I", -1j)], display_format="dense"),
                FermionicOp([("I+I-I+I-", 1j), ("I-I+I-I+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_puccd_ansatz(self, num_spin_orbitals, num_particles, expect):
        """Tests the PUCCD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = PUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    @unpack
    @data(
        (
            4,
            (1, 1),
            (True, True),
            [
                FermionicOp([("+-II", 1j), ("-+II", 1j)], display_format="dense"),
                FermionicOp([("II+-", 1j), ("II-+", 1j)], display_format="dense"),
                FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense"),
            ],
        ),
        (
            4,
            (1, 1),
            (True, False),
            [
                FermionicOp([("+-II", 1j), ("-+II", 1j)], display_format="dense"),
                FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense"),
            ],
        ),
        (
            4,
            (1, 1),
            (False, True),
            [
                FermionicOp([("II+-", 1j), ("II-+", 1j)], display_format="dense"),
                FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_puccd_ansatz_with_singles(
        self, num_spin_orbitals, num_particles, include_singles, expect
    ):
        """Tests the PUCCD Ansatz with included single excitations."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = PUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            include_singles=include_singles,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    def test_raise_non_singlet(self):
        """Test an error is raised when the number of alpha and beta electrons differ."""
        with self.assertRaises(QiskitNatureError):
            PUCCD(num_particles=(2, 1))

    @unpack
    @data(
        (
            6,
            (1, 1),
            [
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+-", 1j), ("I-+I-+", -1j)], display_format="dense"),
            ],
        ),
        (
            6,
            (2, 2),
            [
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+-", 1j), ("I-+I-+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_puccd_ansatz_generalized(self, num_spin_orbitals, num_particles, expect):
        """Tests the generalized PUCCD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = PUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            generalized=True,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)
