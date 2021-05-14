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
from qiskit.opflow import I, Z

from qiskit_nature.problems.sampling.protein_folding.peptide.beads.side_bead import SideBead
from test import QiskitNatureTestCase


class TestSideBead(QiskitNatureTestCase):
    """Tests Side Bead."""

    def test_side_bead_constructor(self):
        """Tests that a SideBead is created."""
        residue_type = "S"
        turn_qubits = [Z, Z]
        side_bead = SideBead(residue_type, turn_qubits)

        indic_0, indic_1, indic_2, indic_3 = side_bead.get_indicator_functions()

        assert indic_0 == 2.0 * I - 2.0 * Z  # TODO why is this not reduced?
        assert indic_1 == 0.0 * I
        assert indic_2 == 0.0 * I
        assert indic_3 == I
