from __future__ import annotations
from abc import abstractmethod, abstractclassmethod, ABC
from typing import Dict, Iterator
import math
from xmlrpc.client import Boolean

from qiskit.opflow.mixins import StarAlgebraMixin
from qiskit.quantum_info.operators.mixins import TolerancesMixin

class SparseLabelOp(StarAlgebraMixin, TolerancesMixin, ABC):
    def __init__(self, data: Dict[str, complex], register_length: int = None):
        self._data = data # stores strings and numbers
        self._register_length = register_length

    def add(self, other) -> SparseLabelOp:
        """Return Operator addition of self and other"""
        new_data = self._data.copy()

        for key, value in other.items():
            if key in new_data.keys():
                new_data[key] += value
            else:
                new_data[key] = value

        return self.__class__(new_data)
    
    def mul(self, other: complex) -> SparseLabelOp:
        """Return scalar multiplication of self and other"""
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        return self.__class__( #Max to think about...
            [(label, coeff * other) for label, coeff in self._data],
            register_length=self._register_length,
        )

    def compose(self, other: SparseLabelOp) -> SparseLabelOp:
        """Composes two ``SparseLabelOp`` instances.

        Args:
            other: another instance of ``SparseLabelOp``.

        Returns:
            Either a zero operator or a new instance of ``SparseLabelOp``.
        
        Raises:
            TypeError: invalid operator type provided
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Compose argument must be of type '{type(self).__name__}', but type '{type(other).__name__}' provided"
            )

        new_data = list(
            filter(
                lambda x: x[1] != 0,
                (
                    (label1 + label2, cf1 * cf2)
                    for label2, cf2 in other._data
                    for label1, cf1 in self._data
                ),
            )
        )
        register_length = max(self._register_length, other._register_length)
        if not new_data:
            return self.__class__.zero(register_length)
        return self.__class__(new_data, register_length)
    
    def adjoint(self) -> SparseLabelOp:
        """Compute the Adjoint of the operator"""
        new_data = {val: val.conjugate() for val in self._data}
        return self.__class__(new_data, self._register_length)

    def __eq__(self, other) -> Boolean:
        """Check equality of two ``SparseLabelOp`` instances
        
        Args:
            other: the second ``SparseLabelOp`` to compare the first with.

        Returns:
            Bool: True if operators are equal, False if not.
        """
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        for key, val in self._data.items():
            if not math.isclose(val, other._data[key], rel_tol=self.rtol, abs_tol=self.atol):
                return False
        return True
    
    def __iter__(self) -> Iterator[SparseLabelOp]:
        """Iterate through SparseLabelOp items"""
        return iter(self._data.items())

    @abstractmethod
    def commutativity(self) -> bool:
        # a*b = b*a OR -b*a, communtativity tells you if +/-1
        # return true if commutes (+1), false if anti-commutes
        # return true default ?
        ...

    @abstractmethod
    def normal_ordered(self) -> SparseLabelOp:
        """Convert to the equivalent operator with normal order.

        Returns a new operator (the original operator is not modified).
        The returned operator is in sparse label mode.

        Returns:
            The normal ordered operator.
        """
        # normal ordered = first all creation then all annhiliation operators
        # AND want creeation to be sorted by index, and an to be sorted by index
        # NOT normal ordered: +_0 -_0 +_1 -_1
# Normal ordered: +_0 +_1 -_0 -_1
# by default retuirns same object maybe... max needs to think about
        ...

    @abstractmethod
    def simplify(self) -> SparseLabelOp:
        """Simplify the operator.

        Merges terms with same labels and eliminates terms with coefficients close to 0.
        Returns a new operator (the original operator is not modified).

        Returns:
            The simplified operator.
        """

    @abstractclassmethod
    def zero(cls, register_length: int) -> SparseLabelOp:
        """Constructs a zero-operator.

        Args:
            register_length: the length of the operator.

        Returns:
            The zero-operator of the given length.
        """

    @abstractclassmethod
    def one(cls, register_length: int) -> SparseLabelOp:
        """Constructs a unity-operator.

        Args:
            register_length: the length of the operator.

        Returns:
            The unity-operator of the given length.
        """