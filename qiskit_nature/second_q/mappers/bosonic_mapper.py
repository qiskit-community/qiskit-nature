# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bosonic Mapper."""

from __future__ import annotations

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.exceptions import QiskitNatureError

from qiskit_nature.second_q.operators import BosonicOp

from .qubit_mapper import ListOrDictType, QubitMapper


class BosonicMapper(QubitMapper):
    """
    Mapper of Bosonic Operator to Qubit Operator

    Workflow:
        First, the user initializes an instance of BosonicLinearMapper, which call the __init__ method 
        defined in this class.
        Then, the user calls BosonicLinearMapper.map(). Since map is not defined in BosonicLinearMapper,
        the call is actually to the map method in this class.
        QubitMapper.map() calls QubitMapper.map() thank to the super() class.
        Inside QubitMapper.map() there is a call to _map_single. This method is present in QubitMapper
        class, but it is overloaded in BosonicLinearMapper. 
        Thus, the call is to BosonicLinearMapper._map_single()
    """

    def __init__(self, truncation: int) -> None:
        super().__init__()
        if (truncation < 1):
            raise QiskitNatureError("Truncation for bosonic linear mapper must be at least 1. " +
                                    f"Detected value: {truncation}")
        self.truncation = truncation

    def map(
        self,
        second_q_ops: BosonicOp | ListOrDictType[BosonicOp],
        *,
        register_length: int | None = None
    ) -> SparsePauliOp | PauliSumOp | ListOrDictType[SparsePauliOp | PauliSumOp]:
        return super().map(second_q_ops, register_length=register_length)
