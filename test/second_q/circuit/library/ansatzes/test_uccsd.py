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

"""Test the UCCSD Ansatz."""

from test import QiskitNatureTestCase
from test.circuit.library.ansatzes.test_ucc import assert_ucc_like_ansatz

from ddt import ddt, data, unpack

from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.operators.fermionic import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators import QubitConverter


@ddt
class TestUCCSD(QiskitNatureTestCase):
    """Tests for the UCCSD Ansatz."""

    @unpack
    @data(
        (
            4,
            (1, 1),
            [
                FermionicOp([("+-II", 1j), ("-+II", 1j)], display_format="dense"),
                FermionicOp([("II+-", 1j), ("II-+", 1j)], display_format="dense"),
                FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense"),
            ],
        ),
        (
            8,
            (2, 2),
            [
                FermionicOp([("+I-IIIII", 1j), ("-I+IIIII", 1j)], display_format="dense"),
                FermionicOp([("+II-IIII", 1j), ("-II+IIII", 1j)], display_format="dense"),
                FermionicOp([("I+-IIIII", 1j), ("I-+IIIII", 1j)], display_format="dense"),
                FermionicOp([("I+I-IIII", 1j), ("I-I+IIII", 1j)], display_format="dense"),
                FermionicOp([("IIII+I-I", 1j), ("IIII-I+I", 1j)], display_format="dense"),
                FermionicOp([("IIII+II-", 1j), ("IIII-II+", 1j)], display_format="dense"),
                FermionicOp([("IIIII+-I", 1j), ("IIIII-+I", 1j)], display_format="dense"),
                FermionicOp([("IIIII+I-", 1j), ("IIIII-I+", 1j)], display_format="dense"),
                FermionicOp([("++--IIII", 1j), ("--++IIII", -1j)], display_format="dense"),
                FermionicOp([("+I-I+I-I", 1j), ("-I+I-I+I", -1j)], display_format="dense"),
                FermionicOp([("+I-I+II-", 1j), ("-I+I-II+", -1j)], display_format="dense"),
                FermionicOp([("+I-II+-I", 1j), ("-I+II-+I", -1j)], display_format="dense"),
                FermionicOp([("+I-II+I-", 1j), ("-I+II-I+", -1j)], display_format="dense"),
                FermionicOp([("+II-+I-I", 1j), ("-II+-I+I", -1j)], display_format="dense"),
                FermionicOp([("+II-+II-", 1j), ("-II+-II+", -1j)], display_format="dense"),
                FermionicOp([("+II-I+-I", 1j), ("-II+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+II-I+I-", 1j), ("-II+I-I+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+I-I", 1j), ("I-+I-I+I", -1j)], display_format="dense"),
                FermionicOp([("I+-I+II-", 1j), ("I-+I-II+", -1j)], display_format="dense"),
                FermionicOp([("I+-II+-I", 1j), ("I-+II-+I", -1j)], display_format="dense"),
                FermionicOp([("I+-II+I-", 1j), ("I-+II-I+", -1j)], display_format="dense"),
                FermionicOp([("I+I-+I-I", 1j), ("I-I+-I+I", -1j)], display_format="dense"),
                FermionicOp([("I+I-+II-", 1j), ("I-I+-II+", -1j)], display_format="dense"),
                FermionicOp([("I+I-I+-I", 1j), ("I-I+I-+I", -1j)], display_format="dense"),
                FermionicOp([("I+I-I+I-", 1j), ("I-I+I-I+", -1j)], display_format="dense"),
                FermionicOp([("IIII++--", 1j), ("IIII--++", -1j)], display_format="dense"),
            ],
        ),
        (
            8,
            (2, 1),
            [
                FermionicOp([("+I-IIIII", 1j), ("-I+IIIII", 1j)], display_format="dense"),
                FermionicOp([("+II-IIII", 1j), ("-II+IIII", 1j)], display_format="dense"),
                FermionicOp([("I+-IIIII", 1j), ("I-+IIIII", 1j)], display_format="dense"),
                FermionicOp([("I+I-IIII", 1j), ("I-I+IIII", 1j)], display_format="dense"),
                FermionicOp([("IIII+-II", 1j), ("IIII-+II", 1j)], display_format="dense"),
                FermionicOp([("IIII+I-I", 1j), ("IIII-I+I", 1j)], display_format="dense"),
                FermionicOp([("IIII+II-", 1j), ("IIII-II+", 1j)], display_format="dense"),
                FermionicOp([("++--IIII", 1j), ("--++IIII", -1j)], display_format="dense"),
                FermionicOp([("+I-I+-II", 1j), ("-I+I-+II", -1j)], display_format="dense"),
                FermionicOp([("+I-I+I-I", 1j), ("-I+I-I+I", -1j)], display_format="dense"),
                FermionicOp([("+I-I+II-", 1j), ("-I+I-II+", -1j)], display_format="dense"),
                FermionicOp([("+II-+-II", 1j), ("-II+-+II", -1j)], display_format="dense"),
                FermionicOp([("+II-+I-I", 1j), ("-II+-I+I", -1j)], display_format="dense"),
                FermionicOp([("+II-+II-", 1j), ("-II+-II+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+-II", 1j), ("I-+I-+II", -1j)], display_format="dense"),
                FermionicOp([("I+-I+I-I", 1j), ("I-+I-I+I", -1j)], display_format="dense"),
                FermionicOp([("I+-I+II-", 1j), ("I-+I-II+", -1j)], display_format="dense"),
                FermionicOp([("I+I-+-II", 1j), ("I-I+-+II", -1j)], display_format="dense"),
                FermionicOp([("I+I-+I-I", 1j), ("I-I+-I+I", -1j)], display_format="dense"),
                FermionicOp([("I+I-+II-", 1j), ("I-I+-II+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_uccsd_ansatz(self, num_spin_orbitals, num_particles, expect):
        """Tests the UCCSD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCCSD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    @unpack
    @data(
        (
            6,
            (1, 1),
            [
                FermionicOp([("+-IIII", 1j), ("-+IIII", 1j)], display_format="dense"),
                FermionicOp([("+I-III", 1j), ("-I+III", 1j)], display_format="dense"),
                FermionicOp([("I+-III", 1j), ("I-+III", 1j)], display_format="dense"),
                FermionicOp([("III+-I", 1j), ("III-+I", 1j)], display_format="dense"),
                FermionicOp([("III+I-", 1j), ("III-I+", 1j)], display_format="dense"),
                FermionicOp([("IIII+-", 1j), ("IIII-+", 1j)], display_format="dense"),
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+-I+I-", 1j), ("-+I-I+", -1j)], display_format="dense"),
                FermionicOp([("+-II+-", 1j), ("-+II-+", -1j)], display_format="dense"),
                FermionicOp([("+I-+-I", 1j), ("-I+-+I", -1j)], display_format="dense"),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
                FermionicOp([("+I-I+-", 1j), ("-I+I-+", -1j)], display_format="dense"),
                FermionicOp([("I+-+-I", 1j), ("I-+-+I", -1j)], display_format="dense"),
                FermionicOp([("I+-+I-", 1j), ("I-+-I+", -1j)], display_format="dense"),
                FermionicOp([("I+-I+-", 1j), ("I-+I-+", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_uccsd_ansatz_generalized(self, num_spin_orbitals, num_particles, expect):
        """Tests the generalized UCCSD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCCSD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            generalized=True,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)

    @unpack
    @data(
        (
            4,
            (1, 1),
            [
                FermionicOp([("+-II", 1j), ("-+II", 1j)], display_format="dense"),
                FermionicOp([("+II-", 1j), ("-II+", 1j)], display_format="dense"),
                FermionicOp([("I-+I", 1j), ("I+-I", 1j)], display_format="dense"),
                FermionicOp([("II+-", 1j), ("II-+", 1j)], display_format="dense"),
                FermionicOp([("+-+-", 1j), ("-+-+", -1j)], display_format="dense"),
            ],
        ),
        (
            6,
            (1, 1),
            [
                FermionicOp([("+-IIII", 1j), ("-+IIII", 1j)], display_format="dense"),
                FermionicOp([("+I-III", 1j), ("-I+III", 1j)], display_format="dense"),
                FermionicOp([("+III-I", 1j), ("-III+I", 1j)], display_format="dense"),
                FermionicOp([("+IIII-", 1j), ("-IIII+", 1j)], display_format="dense"),
                FermionicOp([("I-I+II", 1j), ("I+I-II", 1j)], display_format="dense"),
                FermionicOp([("II-+II", 1j), ("II+-II", 1j)], display_format="dense"),
                FermionicOp([("III+-I", 1j), ("III-+I", 1j)], display_format="dense"),
                FermionicOp([("III+I-", 1j), ("III-I+", 1j)], display_format="dense"),
                FermionicOp([("+--+II", 1j), ("-++-II", -1j)], display_format="dense"),
                FermionicOp([("+-I+-I", 1j), ("-+I-+I", -1j)], display_format="dense"),
                FermionicOp([("+-I+I-", 1j), ("-+I-I+", -1j)], display_format="dense"),
                FermionicOp([("+I-+-I", 1j), ("-I+-+I", -1j)], display_format="dense"),
                FermionicOp([("+I-+I-", 1j), ("-I+-I+", -1j)], display_format="dense"),
                FermionicOp([("+II+--", 1j), ("-II-++", -1j)], display_format="dense"),
            ],
        ),
    )
    def test_uccsd_ansatz_preserve_spin(self, num_spin_orbitals, num_particles, expect):
        """Tests UCCSD Ansatz with spin flips."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCCSD(
            qubit_converter=converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            preserve_spin=False,
        )

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)
