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
"""An abstract class defining a bead of a peptide."""
from abc import ABC
from typing import List, Tuple, Union, Callable

from qiskit.opflow import PauliOp, OperatorBase

from ..pauli_ops_builder import _build_full_identity
from ...residue_validator import _validate_residue_symbol


class BaseBead(ABC):
    """An abstract class defining a bead of a peptide."""

    def __init__(
        self,
        chain_type: str,
        main_index: int,
        residue_type: str,
        turn_qubits: List[PauliOp],
        build_turn_indicator_fun_0: Callable,
        build_turn_indicator_fun_1: Callable,
        build_turn_indicator_fun_2: Callable,
        build_turn_indicator_fun_3: Callable,
    ):
        """
        Args:
            chain_type: Type of the chain, either "main_chain" or "side_chain".
            main_index: index of the bead on the main chain in a peptide.
            residue_type: A character representing the type of a residue for the bead.
            turn_qubits: A list of Pauli operators that encodes the turn following from a
                            given bead index.
            build_turn_indicator_fun_0: method that build turn indicator functions for the bead.
            build_turn_indicator_fun_1: method that build turn indicator functions for the bead.
            build_turn_indicator_fun_2: method that build turn indicator functions for the bead.
            build_turn_indicator_fun_3: method that build turn indicator functions for the bead.
        """
        self.chain_type = chain_type
        self.main_index = main_index
        self._residue_type = residue_type
        _validate_residue_symbol(residue_type)
        self._turn_qubits = turn_qubits
        if self._residue_type is not None and self.turn_qubits is not None:
            full_id = _build_full_identity(turn_qubits[0].num_qubits)
            self._turn_indicator_fun_0 = build_turn_indicator_fun_0(full_id)
            self._turn_indicator_fun_1 = build_turn_indicator_fun_1(full_id)
            self._turn_indicator_fun_2 = build_turn_indicator_fun_2(full_id)
            self._turn_indicator_fun_3 = build_turn_indicator_fun_3(full_id)

    @property
    def turn_qubits(self) -> List[PauliOp]:
        """Returns the list of two qubits that encode the turn following from the bead."""
        return self._turn_qubits

    @property
    def residue_type(self) -> str:
        """Returns a residue type."""
        return self._residue_type

    # for the turn that leads from the bead
    def get_indicator_functions(
        self,
    ) -> Union[None, Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase]]:
        """
        Returns all turn indicator functions for the bead.
        Returns:
            turn_indicator_fun_0, turn_indicator_fun_1, \
               turn_indicator_fun_2, turn_indicator_fun_3: A tuple of all turn indicator
               functions for the bead.
        """
        if self.turn_qubits is None:
            return None
        return (
            self._turn_indicator_fun_0,
            self._turn_indicator_fun_1,
            self._turn_indicator_fun_2,
            self._turn_indicator_fun_3,
        )
