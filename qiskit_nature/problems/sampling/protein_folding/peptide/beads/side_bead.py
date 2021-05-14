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
from typing import List

from qiskit.opflow import PauliOp

from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead


class SideBead(BaseBead):

    def __init__(self, residue_type: str, turn_qubits: List[PauliOp]):
        super().__init__(residue_type, turn_qubits)
