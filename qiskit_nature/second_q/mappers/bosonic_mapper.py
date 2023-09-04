# This code is part of a Qiskit project.
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

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import BosonicOp

from .qubit_mapper import ListOrDictType, QubitMapper


class BosonicMapper(QubitMapper):
    """Mapper of Bosonic Operator to Qubit Operator

    The following attributes can be read and updated once the ``BosonicMapper`` object
    has been constructed.

    """

    def __init__(self, max_occupation: int) -> None:
        """
        Args:
            max_occupation: defines the excitation space of the k-th bosonic state. Together with the
                number of modes required to represent the bosonic operator, it defines the minimum length
                of the qubit register. The minimum value is 1.
        """
        super().__init__()
        self.max_occupation = max_occupation

    @property
    def max_occupation(self) -> int:
        """The maximum occupation of any bosonic state."""
        return self._max_occupation

    @max_occupation.setter
    def max_occupation(self, max_occupation: int) -> None:
        if max_occupation < 1:
            raise ValueError(
                f"The maximum occupation must be at least 1, and not {max_occupation}."
            )
        self._max_occupation = max_occupation

    def map(
        self,
        second_q_ops: BosonicOp | ListOrDictType[BosonicOp],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp | ListOrDictType[SparsePauliOp]:
        return super().map(second_q_ops, register_length=register_length)
