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

"""Test the SUCCD Ansatz."""

from test import QiskitNatureTestCase
from test.second_q.circuit.library.ansatzes.test_ucc import assert_ucc_like_ansatz

from ddt import ddt, data, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.circuit.library import SUCCD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import QubitConverter


@ddt
class TestSUCCD(QiskitNatureTestCase):
    """Tests for the SUCCD Ansatz."""

    @unpack
    @data(
        (4, (1, 1), [FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense")]),
        (
            8,
            (2, 2),
            [
                FermionicOp([("+I-I+I-I", 1j), ("-I+I-I+I", -1j)], display_format="dense"),
                FermionicOp([("+I-I+II-", 1j), ("-I+I-II+", -1j)], display_format="dense"),
                FermionicOp([("+I-II+-I", 1j), ("-I+II-+I", -1j)], display_format="dense"),
                FermionicOp([("+I-II+I-", 1j), ("-I+II-I+", -1j)], display_format="dense"),
                FermionicOp([("+II-+II-", 1j), ("-II+-II+", -1j)], display_format="dense"),
                FermionicOp([("+II-I+-I", 1j), ("-II+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+II-I+I-", 1j), ("-II+I-I+", -1j)], display_format="dense"),
                FermionicOp([("I+-II+-I", 1j), ("I-+II-+I", -1j)], display_format="dense"),
                FermionicOp([("I+-II+I-", 1j), ("I-+II-I+", -1j)], display_format="dense"),
                FermionicOp([("I+I-I+I-", 1j), ("I-I+I-I+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_succd_ansatz(self, num_spin_orbitals, num_particles, expect):
        """Tests the SUCCD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(
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
    def test_succd_ansatz_with_singles(
        self, num_spin_orbitals, num_particles, include_singles, expect
    ):
        """Tests the SUCCD Ansatz with included single excitations."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            include_singles=include_singles,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    def test_raise_non_singlet(self):
        """Test an error is raised when the number of alpha and beta electrons differ."""
        with self.assertRaises(QiskitNatureError):
            SUCCD(num_particles=(2, 1))

    @unpack
    @data(
        (
            6,
            (1, 1),
            [
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+-I+I-", 1j), ("-+I-I+", -1j)], display_format="dense"),
                FermionicOp([("+-II+-", 1j), ("-+II-+", -1j)], display_format="dense"),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
                FermionicOp([("+I-I+-", 1j), ("-I+I-+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+-", 1j), ("I-+I-+", -1j)], display_format="dense"),
            ],
        ),
        (
            6,
            (2, 2),
            [
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+-I+I-", 1j), ("-+I-I+", -1j)], display_format="dense"),
                FermionicOp([("+-II+-", 1j), ("-+II-+", -1j)], display_format="dense"),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
                FermionicOp([("+I-I+-", 1j), ("-I+I-+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+-", 1j), ("I-+I-+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_puccd_ansatz_generalized(self, num_spin_orbitals, num_particles, expect):
        """Tests the generalized SUCCD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            generalized=True,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    @unpack
    @data(
        (
            6,
            (1, 1),
            [
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp(
                    [("+-I+I-", 1j), ("-+I-I+", -1j), ("+I-+-I", 1j), ("-I+-+I", -1j)],
                    display_format="dense",
                ),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_succ_mirror(self, num_spin_orbitals, num_particles, expect):
        """Tests the `mirror` option of the SUCCD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            mirror=True,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    @unpack
    @data(
        (
            6,
            (1, 1),
            (True, True),
            [
                FermionicOp([("+-IIII", 1j), ("-+IIII", 1j)], display_format="dense"),
                FermionicOp([("+I-III", 1j), ("-I+III", 1j)], display_format="dense"),
                FermionicOp([("III+-I", 1j), ("III-+I", 1j)], display_format="dense"),
                FermionicOp([("III+I-", 1j), ("III-I+", 1j)], display_format="dense"),
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", (-0 - 1j))], display_format="dense"),
                FermionicOp(
                    [("+-I+I-", 1j), ("-+I-I+", (-0 - 1j)), ("+I-+-I", 1j), ("-I+-+I", (-0 - 1j))],
                    display_format="dense",
                ),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", (-0 - 1j))], display_format="dense"),
            ],
        ),
        (
            6,
            (1, 1),
            (True, False),
            [
                FermionicOp([("+-IIII", 1j), ("-+IIII", 1j)], display_format="dense"),
                FermionicOp([("+I-III", 1j), ("-I+III", 1j)], display_format="dense"),
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", (-0 - 1j))], display_format="dense"),
                FermionicOp(
                    [("+-I+I-", 1j), ("-+I-I+", (-0 - 1j)), ("+I-+-I", 1j), ("-I+-+I", (-0 - 1j))],
                    display_format="dense",
                ),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", (-0 - 1j))], display_format="dense"),
            ],
        ),
        (
            6,
            (1, 1),
            (False, True),
            [
                FermionicOp([("III+-I", 1j), ("III-+I", 1j)], display_format="dense"),
                FermionicOp([("III+I-", 1j), ("III-I+", 1j)], display_format="dense"),
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", (-0 - 1j))], display_format="dense"),
                FermionicOp(
                    [("+-I+I-", 1j), ("-+I-I+", (-0 - 1j)), ("+I-+-I", 1j), ("-I+-+I", (-0 - 1j))],
                    display_format="dense",
                ),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", (-0 - 1j))], display_format="dense"),
            ],
        ),
    )
    def test_succ_mirror_with_singles(
        self, num_spin_orbitals, num_particles, include_singles, expect
    ):
        """Tests the succ_mirror Ansatz with included single excitations."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            include_singles=include_singles,
            mirror=True,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    @unpack
    @data(
        (
            6,
            (1, 1),
            [
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", (-0 - 1j))], display_format="dense"),
                FermionicOp(
                    [("+-I+I-", 1j), ("-+I-I+", (-0 - 1j)), ("+I-+-I", 1j), ("-I+-+I", (-0 - 1j))],
                    display_format="dense",
                ),
                FermionicOp(
                    [("+-II+-", 1j), ("-+II-+", (-0 - 1j)), ("I+-+-I", 1j), ("I-+-+I", (-0 - 1j))],
                    display_format="dense",
                ),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", (-0 - 1j))], display_format="dense"),
                FermionicOp(
                    [("+I-I+-", 1j), ("-I+I-+", (-0 - 1j)), ("I+-+I-", 1j), ("I-+-I+", (-0 - 1j))],
                    display_format="dense",
                ),
                FermionicOp([("I+-I+-", 1j), ("I-+I-+", (-0 - 1j))], display_format="dense"),
            ],
        )
    )
    def test_succ_mirror_ansatz_generalized(self, num_spin_orbitals, num_particles, expect):
        """Tests the generalized succ_mirror Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            generalized=True,
            mirror=True,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)
