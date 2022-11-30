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

"""Spin Mapper."""

from abc import abstractmethod
from typing import Union

from qiskit.opflow import PauliSumOp
from qiskit_nature.second_q.operators import SpinOp

from .qubit_mapper import QubitMapper, _ListOrDict


class SpinMapper(QubitMapper):
    """Mapper of Spin Operator to Qubit Operator"""

    @abstractmethod
    def map(self, second_q_op: SpinOp) -> PauliSumOp:
        """Maps a :class:`~qiskit_nature.second_q.operators.SpinOp` to a `PauliSumOp`.

        Args:
            second_q_op: the `SpinOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()

    def convert_match(
        self,
        second_q_ops: Union[SpinOp, _ListOrDict[SpinOp]],
        *,
        suppress_none: bool = False,
        check_commutes: bool = True,
    ) -> _ListOrDict[PauliSumOp]:
        """A convenience method to map second quantized operators based on current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof
            sort_operators: Boolean

        Returns:
            A qubit operator in the form of a PauliSumOp, or list thereof if a list of
            second quantized operators was supplied
        """
        wrapped_type = type(second_q_ops)

        if isinstance(second_q_ops, SpinOp):
            second_q_ops = [second_q_ops]
            suppress_none = False  # When only a single op we will return None back

        wrapped_second_q_ops: _ListOrDict[SpinOp] = _ListOrDict(second_q_ops)

        qubit_ops = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = self.map(second_q_op)

        returned_ops: Union[PauliSumOp, ListOrDictType[PauliSumOp]]

        if issubclass(wrapped_type, SpinOp):
            returned_ops = list(iter(qubit_ops))[0][1]
        elif wrapped_type == list:
            if suppress_none:
                returned_ops = [op for _, op in iter(qubit_ops) if op is not None]
            else:
                returned_ops = [op for _, op in iter(qubit_ops)]
        elif wrapped_type == dict:
            returned_ops = dict(iter(qubit_ops))

        return returned_ops
