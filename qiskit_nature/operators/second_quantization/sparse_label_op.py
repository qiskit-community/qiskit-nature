from __future__ import annotations
from abc import abstractmethod, abstractclassmethod, ABC
from typing import Dict

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
            register_length=self.register_length,
        )

    def compose(self, other: SparseLabelOp) -> SparseLabelOp:
        if not isinstance(other, SparseLabelOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )

        new_data = #Max to think about... something with commutativity
        register_length = max(self.register_length, other.register_length)
        if not new_data:
            return self.__class__.zero(register_length)
        return self.__class__(new_data, register_length)
    
    def adjoint(self) -> SparseLabelOp:
        return # Max to think about...

    @abstractmethod
    def commutativity(self) -> bool:
        # a*b = b*a OR -b*a, communtativity tells you if +/-1
        # return true if commutes (+1), false if anti-commutes
        # return true default
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