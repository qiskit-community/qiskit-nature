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

"""A converter from Second-Quantized to Qubit Operators."""
from typing import List, Optional, Tuple, Union

from qiskit.opflow import PauliSumOp
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature import QiskitNatureError
from qiskit_nature.mappers.second_quantization import QubitMapper

from . import SecondQuantizedOp


class QubitConverter:
    """A converter from Second-Quantized to Qubit Operators."""

    def __init__(self,
                 mappers: Union[QubitMapper, Tuple[QubitMapper, QubitMapper]],
                 z2symmetry_reduction: Optional[List[int]] = None):

        self._mappers = mappers
        self._z2symmetry_reduction = z2symmetry_reduction
        self._z2symmetries = None

    @property
    def z2symmetry_reduction(self) -> Optional[List[int]]:
        """Get z2symmetry_reduction"""
        return self._z2symmetry_reduction

    @z2symmetry_reduction.setter
    def z2symmetry_reduction(self, z2symmetry_reduction: Optional[List[int]]) -> None:
        """Set z2symmetry_reduction"""
        self._z2symmetry_reduction = z2symmetry_reduction
        self._z2symmetries = None

    @property
    def z2symmetries(self) -> Optional[Z2Symmetries]:
        """Get z2symmetries. Will be `None` until :meth:`to_qubit_ops` has been called."""
        return self._z2symmetries

    def to_qubit_ops(self, second_q_ops: List[SecondQuantizedOp],
                     z2symmetry_reduction: List[int] = None,
                     ) -> List[Optional[PauliSumOp]]:
        """
        Maps the given list of second quantized operators to qubit operators. Also it will
        carry out z2 symmetry reduction on the qubit operators if z2symmetry_reduction has
        been specified.

        Args:
            second_q_ops: The list of second quantized operators to be converted
            z2symmetry_reduction: Optional z2symmetry reduction, the sector of the symmetry

        Returns:
            A list of qubit operators or the same length as the second_q_ops list. The first
            operator in the second_q_ops list is treated as the main operator and others must
            commute with its symmetry, when symmetry reduction is being done. If it does not
            then the position in the output list will be set to `None` to preserve the order.
        """
        if z2symmetry_reduction is not None:
            self.z2symmetry_reduction = z2symmetry_reduction

        qubit_ops = self._map_to_qubits(second_q_ops)
        qubit_ops_reduced = self._symmetry_reduce(qubit_ops)

        return qubit_ops_reduced

    def _map_to_qubits(self, second_q_ops: List[SecondQuantizedOp]) -> List[PauliSumOp]:
        if isinstance(self._mappers, tuple):
            raise NotImplementedError

        qubit_ops: List[PauliSumOp] = []
        main_op = second_q_ops[0]
        if main_op._fermion is not None:
            for op in second_q_ops:
                if op._fermion is None:
                    raise QiskitNatureError("Second quantized operators must contain same type")
                qubit_ops.append(self._mappers.map(op._fermion))
        else:
            raise QiskitNatureError("Cannot map invalid second quantized operator")

        return qubit_ops

    def _symmetry_reduce(self, qubit_ops: List[PauliSumOp]) -> List[Optional[PauliSumOp]]:

        if self.z2symmetry_reduction is None:
            reduced_qubit_ops = qubit_ops
            self._z2symmetries = Z2Symmetries([], [], [], None)
        else:
            # We have a symmetry sector so we can use it to reduce the operators.
            reduced_qubit_ops = qubit_ops  # TODO actually reduce the operators...
            self._z2symmetries = Z2Symmetries([], [], [], None)  # and set the actual symmetries

        return reduced_qubit_ops
