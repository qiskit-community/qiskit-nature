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
from abc import ABC
from typing import List, Tuple

from qiskit.opflow import PauliOp, OperatorBase

from problems.sampling.protein_folding.exceptions.invalid_residue_exception import \
    InvalidResidueException
from problems.sampling.protein_folding.residue_validator import _validate_residue_symbol


class BaseBead(ABC):

    def __init__(self, residue_type: str, turn_qubits: List[PauliOp]):

        self._residue_type = residue_type
        _validate_residue_symbol(residue_type)
        self._turn_qubits = turn_qubits

    @property
    def turn_qubits(self):
        return self._turn_qubits

    @property
    def residue_type(self):
        return self._residue_type

    # for the turn that leads from the bead
    def get_indicator_functions(self) -> Tuple[
        OperatorBase, OperatorBase, OperatorBase, OperatorBase]:
        if self.turn_qubits is None:
            return None
        return self._indic_0, self._indic_1, self._indic_2, self._indic_3
