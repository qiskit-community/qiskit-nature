# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
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
from typing import TypeVar, Dict, Iterable, Generic, Generator

from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_algorithms.list_or_dict import ListOrDict as ListOrDictType

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

    @abstractmethod
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
