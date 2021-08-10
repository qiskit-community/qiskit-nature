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
from typing import Tuple, Union, Callable, Optional

from qiskit.opflow import PauliOp, OperatorBase

from ..pauli_ops_builder import _build_full_identity
from ...residue_validator import _validate_residue_symbol


class BaseBead(ABC):
    """An abstract class defining a bead of a peptide."""

    def __init__(
        self,
        chain_type: str,
        main_index: int,
        residue_type: Optional[str],
        turn_qubits: Tuple[PauliOp, PauliOp],
        build_turn_indicator_fun_0: Callable[[], OperatorBase],
        build_turn_indicator_fun_1: Callable[[], OperatorBase],
        build_turn_indicator_fun_2: Callable[[], OperatorBase],
        build_turn_indicator_fun_3: Callable[[], OperatorBase],
    ):
        """
        Args:
            chain_type: Type of the chain, either "main_chain" or "side_chain".
            main_index: index of the bead on the main chain in a peptide.
            residue_type: A character representing the type of a residue for the bead. "" in
                        case of non-existing side bead.
            turn_qubits: A tuple of two of Pauli operators that encodes the turn following from a
                            given bead index.
            build_turn_indicator_fun_0: method that build turn indicator functions for the bead.
                                        It is passed by a child class (SideBead or MainBead) and
                                        uses turn qubits to construct a corresponding turn
                                        indicator function (for details, see the paper: paper
                                        Robert et al., npj quantum information 7, 38, 2021).
            build_turn_indicator_fun_1: method that build turn indicator functions for the bead.
                                        It is passed by a child class (SideBead or MainBead) and
                                        uses turn qubits to construct a corresponding turn
                                        indicator function (for details, see the paper: paper
                                        Robert et al., npj quantum information 7, 38, 2021).
            build_turn_indicator_fun_2: method that build turn indicator functions for the bead.
                                        It is passed by a child class (SideBead or MainBead) and
                                        uses turn qubits to construct a corresponding turn
                                        indicator function (for details, see the paper: paper
                                        Robert et al., npj quantum information 7, 38, 2021).
            build_turn_indicator_fun_3: method that build turn indicator functions for the bead.
                                        It is passed by a child class (SideBead or MainBead) and
                                        uses turn qubits to construct a corresponding turn
                                        indicator function (for details, see the paper: paper
                                        Robert et al., npj quantum information 7, 38, 2021).
        """
        self.chain_type = chain_type
        self.main_index = main_index
        self._residue_type = residue_type
        _validate_residue_symbol(residue_type)
        self._turn_qubits = turn_qubits
        if self._residue_type and self.turn_qubits is not None:
            self._full_id = _build_full_identity(turn_qubits[0].num_qubits)
            self._turn_indicator_fun_0 = build_turn_indicator_fun_0()
            self._turn_indicator_fun_1 = build_turn_indicator_fun_1()
            self._turn_indicator_fun_2 = build_turn_indicator_fun_2()
            self._turn_indicator_fun_3 = build_turn_indicator_fun_3()

    @property
    def turn_qubits(self) -> Tuple[PauliOp, PauliOp]:
        """Returns the list of two qubits that encode the turn following from the bead."""
        return self._turn_qubits

    @property
    def residue_type(self) -> Optional[str]:
        """Returns a residue type."""
        return self._residue_type

    # for the turn that leads from the bead
    @property
    def indicator_functions(
        self,
    ) -> Union[None, Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase]]:
        """
        Returns all turn indicator functions for the bead.
        Returns:
            A tuple of all turn indicator functions for the bead.
        """
        if self.turn_qubits is None:
            return None
        return (
            self._turn_indicator_fun_0,
            self._turn_indicator_fun_1,
            self._turn_indicator_fun_2,
            self._turn_indicator_fun_3,
        )
