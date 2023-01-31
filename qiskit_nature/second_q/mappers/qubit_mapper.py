# This code is part of Qiskit.
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

"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Union, TypeVar, Dict, Iterable, Generic, Tuple, Generator, Optional

import numpy as np
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit.algorithms.list_or_dict import ListOrDict as ListOrDictType

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators import SparseLabelOp

# pylint: disable=invalid-name
T = TypeVar("T")


class _ListOrDict(Dict, Iterable, Generic[T]):
    """The ListOrDict utility class.

    This is a utility which allows seamless iteration of a `list` or `dict` object.
    """

    def __init__(self, values: Optional[ListOrDictType] = None):
        """
        Args:
            values: an optional object of `list` or `dict` type.
        """
        if isinstance(values, list):
            values = dict(enumerate(values))
        elif values is None:
            values = {}
        super().__init__(values)

    def __iter__(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """Return the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """Return generator method iterating the contents of this class.

        This generator yields the `(key, value)` pairs of the underlying dictionary. If this object
        was constructed from a list, the keys in this generator are simply the numeric indices.

        This generator also supports overriding the yielded value upon receiving any value other
        than `None` from a `send` [1] instruction.

        [1]: https://docs.python.org/3/reference/expressions.html#generator.send
        """
        for key, value in self.items():
            new_value = yield (key, value)
            if new_value is not None:
                self[key] = new_value

    def unwrap(self, wrapped_type: type, suppress_none: bool = False) -> Dict | Iterable | T:
        """Return the content of this class according to the initial type of the data before
        the creation of the ListOrDict object.

        Args:
            wrapped_type: Type of the data before the creation of the ListOrDict object.
            suppress_none: If None values should be suppressed from the output.

        Returns:
            Content of the current class instance as a list, a dictionary or a single element.
        """
        if wrapped_type == list:
            if suppress_none:
                return [op for _, op in iter(self) if op is not None]
            else:
                return [op for _, op in iter(self)]
        if wrapped_type == dict:
            if suppress_none:
                return {key: op for key, op in iter(self) if op is not None}
            else:
                return dict(iter(self))
        # only other case left is that it was a single operator to begin with:
        return list(iter(self))[0][1]


class QubitMapper(ABC):
    """The interface for implementing methods which map from a `SparseLabelOp` to a
    qubit operator in the form of a `PauliSumOp`.
    """

    def __init__(self, allows_two_qubit_reduction: bool = False):
        """
        Args:
            allows_two_qubit_reduction: Set if mapper will create known symmetry such that the
                number of qubits in the mapped operator can be reduced accordingly.
        """
        self._allows_two_qubit_reduction = allows_two_qubit_reduction

    @property
    def allows_two_qubit_reduction(self) -> bool:
        """
        Getter for symmetry information for two qubit reduction

        Returns: If mapping generates the known symmetry that allows two qubit reduction.

        """
        return self._allows_two_qubit_reduction

    @abstractmethod
    def _map_single(self, second_q_op: SparseLabelOp) -> PauliSumOp:
        """Maps a :class:`~qiskit_nature.second_q.operators.SparseLabelOp`
        to a `PauliSumOp`.

        Args:
            second_q_op: the `SparseLabelOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()

    def map(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
        suppress_none: bool = False,
    ) -> PauliSumOp | ListOrDictType[PauliSumOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof.
            suppress_none: If the output list or dict should contain None at positions where an operator
                did not commute with symmetry, to maintain order, or whether that should
                be suppressed. If True the output list or dict length may then be smaller than the input.

        Returns:
            A qubit operator in the form of a PauliSumOp, or list (resp. dict) thereof if a list
            (resp. dict) of second quantized operators was supplied.
        """
        wrapped_type = type(second_q_ops)

        if issubclass(wrapped_type, SparseLabelOp):
            second_q_ops = [second_q_ops]
            suppress_none = False

        wrapped_second_q_ops: _ListOrDict[SparseLabelOp] = _ListOrDict(second_q_ops)

        qubit_ops: _ListOrDict = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = self._map_single(second_q_op)
        returned_ops: Union[PauliSumOp, ListOrDictType[PauliSumOp]] = qubit_ops.unwrap(
            wrapped_type, suppress_none=suppress_none
        )
        return returned_ops

    @classmethod
    @lru_cache(maxsize=32)
    def pauli_table(cls, nmodes: int) -> list[tuple[Pauli, Pauli]]:
        """Generates a Pauli-lookup table mapping from modes to pauli pairs.

        The generated table is processed by :meth:`.QubitMapper.sparse_pauli_operators`.

        Args:
            nmodes: the number of modes for which to generate the table.

        Returns:
            A list of tuples in which the first and second Pauli operator the real and imaginary
            Pauli strings, respectively.
        """

    @classmethod
    @lru_cache(maxsize=32)
    def sparse_pauli_operators(cls, nmodes: int) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        """Generates the cached :class:`.SparsePauliOp` terms.

        This uses :meth:`.QubitMapper.pauli_table` to construct a list of operators used to
        translate the second-quantization symbols into qubit operators.

        Args:
            nmodes: the number of modes for which to generate the operators.

        Returns:
            Two lists stored in a tuple, consisting of the creation and annihilation  operators,
            applied on the individual modes.
        """
        times_creation_op = []
        times_annihilation_op = []

        for paulis in cls.pauli_table(nmodes):
            real_part = SparsePauliOp(paulis[0], coeffs=[0.5])
            imag_part = SparsePauliOp(paulis[1], coeffs=[0.5j])

            # The creation operator is given by 0.5*(X - 1j*Y)
            creation_op = real_part - imag_part
            times_creation_op.append(creation_op)

            # The annihilation operator is given by 0.5*(X + 1j*Y)
            annihilation_op = real_part + imag_part
            times_annihilation_op.append(annihilation_op)

        return (times_creation_op, times_annihilation_op)

    # TODO: remove nmodes argument once having access to SparseLabelOp.register_length
    @classmethod
    def mode_based_mapping(cls, second_q_op: SparseLabelOp, nmodes: int) -> PauliSumOp:
        """Utility method to map a `SparseLabelOp` to a `PauliSumOp` using a pauli table.

        Args:
            second_q_op: the `SparseLabelOp` to be mapped.
            nmodes: the number of modes for which to generate the operators.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: If number length of pauli table does not match the number
                of operator modes, or if the operator has unexpected label content
        """
        times_creation_op, times_annihilation_op = cls.sparse_pauli_operators(nmodes)

        # make sure ret_op_list is not empty by including a zero op
        ret_op_list = [SparsePauliOp("I" * nmodes, coeffs=[0])]

        for terms, coeff in second_q_op.terms():
            # 1. Initialize an operator list with the identity scaled by the `coeff`
            ret_op = SparsePauliOp("I" * nmodes, coeffs=np.array([coeff]))

            # Go through the label and replace the fermion operators by their qubit-equivalent, then
            # save the respective Pauli string in the pauli_str list.
            for term in terms:
                char = term[0]
                if char == "":
                    break
                position = int(term[1])
                if char == "+":
                    ret_op = ret_op.compose(times_creation_op[position], front=True)
                elif char == "-":
                    ret_op = ret_op.compose(times_annihilation_op[position], front=True)
                # catch any disallowed labels
                else:
                    raise QiskitNatureError(
                        f"FermionicOp label included '{char}'. Allowed characters: I, N, E, +, -"
                    )
            ret_op_list.append(ret_op)

        return PauliSumOp(SparsePauliOp.sum(ret_op_list).simplify())
