# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic Mapper."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import FermionicOp

from .qubit_mapper import ListOrDictType, QubitMapper


class FermionicMapper(QubitMapper):
    """Mapper of Fermionic Operator to Qubit Operator"""

    def map(
        self,
        second_q_ops: FermionicOp | ListOrDictType[FermionicOp],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp | ListOrDictType[SparsePauliOp]:
        return super().map(second_q_ops, register_length=register_length)
