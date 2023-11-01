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

"""Qubit Mapper interface."""

from __future__ import annotations

from abc import ABC
from functools import lru_cache
from typing import TypeVar, Dict, Iterable, Generic, Generator

import numpy as np
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.list_or_dict import ListOrDict as ListOrDictType

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators import SparseLabelOp

# pylint: disable=invalid-name
T = TypeVar("T")


class _ListOrDict(Dict, Iterable, Generic[T]):
    """The ListOrDict utility class.

    This is a utility which allows seamless iteration of a `list` or `dict` object.
    """

    def __init__(self, values: ListOrDictType | None = None):
        """
        Args:
            values: an optional object of `list` or `dict` type.
        """
        if isinstance(values, list):
            values = dict(enumerate(values))
        elif values is None:
            values = {}
        super().__init__(values)

    def __iter__(self) -> Generator[tuple[int | str, T], T, None]:
        """Return the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[tuple[int | str, T], T, None]:
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

    @classmethod
    def wrap(cls, objects: dict | list | T) -> tuple[_ListOrDict, type]:
        """Wraps the provided objects into a ``_ListOrDict`` instance.

        Args:
            objects: a dict, list or single object instance to be wrapped.

        Returns:
            A tuple consisting of the constructed ``_ListOrDict`` and the original type that was
            wrapped.
        """
        wrapped_type = type(objects)

        if not issubclass(wrapped_type, (dict, list)):
            objects = [objects]

        wrapped_objects = cls(objects)

        return wrapped_objects, wrapped_type

    def unwrap(self, wrapped_type: type, *, suppress_none: bool = True) -> dict | Iterable | T:
        """Return the content of this class according to the initial type of the data before
        the creation of the ListOrDict object.

        Args:
            wrapped_type: Type of the data before the creation of the ListOrDict object.
            suppress_none: If None values should be suppressed from the output.

        Returns:
            Content of the current class instance as a list, a dictionary or a single element.
        """

        def _qubit_op_type_wrapper(qubit_op: SparsePauliOp | None):
            if qubit_op is None:
                return None
            return qubit_op

        if wrapped_type == list:
            if suppress_none:
                return [_qubit_op_type_wrapper(op) for _, op in iter(self) if op is not None]
            else:
                return [_qubit_op_type_wrapper(op) for _, op in iter(self)]
        if wrapped_type == dict:
            if suppress_none:
                return {key: _qubit_op_type_wrapper(op) for key, op in iter(self) if op is not None}
            else:
                return {key: _qubit_op_type_wrapper(op) for key, op in iter(self)}
        # only other case left is that it was a single operator to begin with:
        return _qubit_op_type_wrapper(list(iter(self))[0][1])


class QubitMapper(ABC):
    """The interface for implementing methods which map from a ``SparseLabelOp`` to a
    qubit operator in the form of a ``SparsePauliOp``.
    """

    def _map_single(
        self, second_q_op: SparseLabelOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        """Maps a :class:`~qiskit_nature.second_q.operators.SparseLabelOp`
        to a ``SparsePauliOp``.

        Args:
            second_q_op: the ``SparseLabelOp`` to be mapped.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the operator being mapped. This is possible because the
                ``register_length`` is considered a lower bound in a ``SparseLabelOp``.

        Returns:
            The qubit operator corresponding to the problem-Hamiltonian in the qubit space.
        """
        return self.mode_based_mapping(second_q_op, register_length=register_length)

    def map(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp | ListOrDictType[SparsePauliOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the ``SparseLabelOp`` being mapped. This is possible because the
                ``register_length`` is considered a lower bound in a ``SparseLabelOp``.

        Returns:
            A qubit operator in the form of a ``SparsePauliOp``, or list (resp. dict) thereof if a
            list (resp. dict) of second quantized operators was supplied.
        """
        wrapped_second_q_ops, wrapped_type = _ListOrDict.wrap(second_q_ops)

        qubit_ops: _ListOrDict = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = self._map_single(second_q_op, register_length=register_length)

        returned_ops = qubit_ops.unwrap(wrapped_type)
        # Note the output of the mapping will never be None for standard mappers other than the
        # TaperedQubitMapper.
        return returned_ops

    @classmethod
    @lru_cache(maxsize=32)
    def pauli_table(cls, register_length: int) -> list[tuple[Pauli, Pauli]]:
        """Generates a Pauli-lookup table mapping from modes to pauli pairs.

        The generated table is processed by :meth:`.QubitMapper.sparse_pauli_operators`.

        Args:
            register_length: the register length for which to generate the table.

        Returns:
            A list of tuples in which the first and second Pauli operator the real and imaginary
            Pauli strings, respectively.
        """

    @classmethod
    @lru_cache(maxsize=32)
    def sparse_pauli_operators(
        cls, register_length: int
    ) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        # pylint: disable=unused-argument
        """Generates the cached :class:`.SparsePauliOp` terms.

        This uses :meth:`.QubitMapper.pauli_table` to construct a list of operators used to
        translate the second-quantization symbols into qubit operators.

        Args:
            register_length: the register length for which to generate the operators.

        Returns:
            Two lists stored in a tuple, consisting of the creation and annihilation  operators,
            applied on the individual modes.
        """
        times_creation_op = []
        times_annihilation_op = []

        for paulis in cls.pauli_table(register_length):
            real_part = SparsePauliOp(paulis[0], coeffs=[0.5])
            imag_part = SparsePauliOp(paulis[1], coeffs=[0.5j])

            # The creation operator is given by 0.5*(X - 1j*Y)
            creation_op = real_part - imag_part
            times_creation_op.append(creation_op)

            # The annihilation operator is given by 0.5*(X + 1j*Y)
            annihilation_op = real_part + imag_part
            times_annihilation_op.append(annihilation_op)

        return (times_creation_op, times_annihilation_op)

    @classmethod
    def mode_based_mapping(
        cls,
        second_q_op: SparseLabelOp,
        register_length: int | None = None,
    ) -> SparsePauliOp:
        # pylint: disable=unused-argument
        """Utility method to map a ``SparseLabelOp`` to a qubit operator using a pauli table.

        Args:
            second_q_op: the `SparseLabelOp` to be mapped.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the operator being mapped. This is possible because the
                ``register_length`` is considered a lower bound.

        Returns:
            The qubit operator corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: If number length of pauli table does not match the number
                of operator modes, or if the operator has unexpected label content
        """
        if register_length is None:
            register_length = second_q_op.register_length

        times_creation_op, times_annihilation_op = cls.sparse_pauli_operators(register_length)

        # make sure ret_op_list is not empty by including a zero op
        ret_op_list = [SparsePauliOp("I" * register_length, coeffs=[0])]

        for terms, coeff in second_q_op.terms():
            # 1. Initialize an operator list with the identity scaled by the `coeff`
            ret_op = SparsePauliOp("I" * register_length, coeffs=np.array([coeff]))

            # Go through the label and replace the fermion operators by their qubit-equivalent, then
            # save the respective Pauli string in the pauli_str list.
            for term in terms:
                char = term[0]
                if char == "":
                    break
                position = int(term[1])
                if char == "+":
                    ret_op = ret_op.compose(times_creation_op[position], front=True).simplify()
                elif char == "-":
                    ret_op = ret_op.compose(times_annihilation_op[position], front=True).simplify()
                # catch any disallowed labels
                else:
                    raise QiskitNatureError(
                        f"FermionicOp label included '{char}'. Allowed characters: I, N, E, +, -"
                    )
            ret_op_list.append(ret_op)

        sparse_op = SparsePauliOp.sum(ret_op_list).simplify()
        return sparse_op
