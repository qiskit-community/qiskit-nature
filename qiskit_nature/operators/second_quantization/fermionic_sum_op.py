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

"""The Fermionic-particle Operator."""

import copy
import functools
import numbers

import numpy as np

from .primitives.fermionic_operator import FermionicOperator
from .sum_op import SumOp


class FermionicSumOp(SumOp):
    """
    Fermionic type operators

    The abstract fermionic registers are implemented in two subclasses, FermionicOperator and
    FermionicSumOp, inspired by the implementation of Pauli operators in qiskit. A
    FermionicOperator is the equivalent of a single Pauli string on a qubit register.
    The FermionicSumOp represents a sum of multiple FermionicOperators. They act on fermionic
    registers of a fixed length determined at the time of initialization.
    """

    def __init__(self, operator_list, register_length=None):

        # 0. Initialize member variables
        if not any(True for _ in operator_list):
            # Treat case of zero operator (empty operator_list)
            assert isinstance(register_length, int), \
                'When instantiating the zero FermionicSumOp, a register length must be provided.'
            self._register_length = register_length
        else:
            # Treat case of nonzero operator_list
            self._register_length = copy.deepcopy(len(operator_list[0]))

        self._operator_dict = {}

        # Go through all operators in the operator list
        for base_operator in operator_list:
            # 1.  Parse
            # 1.1 Check if they are valid, compatible FermionicOperator instances
            assert isinstance(base_operator, FermionicOperator), \
                'FermionicOperators must be built up from `FermionicOperator` objects'
            assert len(base_operator) == self._register_length, \
                'FermionicOperators must act on fermionic registers of same length.'

            # 2.  Add the valid operator to self._operator_dict
            # 2.1 If the operator has zero coefficient, skip the rest of the steps
            if base_operator.coeff == 0:
                continue

            # 2.2 For nonzero coefficient, add the operator the the dictionary of operators
            operator_label = base_operator.label
            if operator_label not in self._operator_dict.keys():
                # If an operator of the same signature (label) is not yet present in
                # self._operator_dict, add it.
                self._operator_dict[operator_label] = base_operator
            else:
                # Else if an operator of the same signature exists already, add the coefficients.
                self._operator_dict[operator_label].coeff += base_operator.coeff

                # If after addition the coefficient is 0, remove the operator from the list
                if self._operator_dict[operator_label].coeff == 0:
                    self._operator_dict.pop(operator_label)

        # 3. Set the particle type
        SumOp.__init__(self, particle_type='fermionic')

    def __repr__(self):
        """Sets the representation of `self` in the console."""

        # 1. Treat the case of the zero-operator:
        if self.operator_list == []:
            return 'zero operator ({})'.format(self.register_length)

        # 2. Treat the general case:
        full_str = ''
        for operator in self.operator_list:
            full_str += '{1} \t {0}\n'.format(operator.coeff, operator.label)
        return full_str

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type, a FermionicOperator or a FermionicSumOp.
        """
        # Catch the case of a zero FermionicSumOp (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, FermionicOperator):
                assert self._register_length == len(other), \
                    'Operators act on Fermion Registers of different length'
            elif isinstance(other, FermionicSumOp):
                assert self._register_length == other._register_length, \
                    'Operators act on Fermion Registers of different length'
            # return FermionicOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, (numbers.Number, FermionicOperator)):
            # Create copy of the FermionicSumOp in which every FermionicOperator is
            # multiplied by `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other
                                for base_operator in self.operator_list]
            return FermionicSumOp(new_operatorlist)

        if isinstance(other, FermionicSumOp):
            # Initialize new operator_list for the returned Fermionic operator
            new_operatorlist = []

            # Catch the case of a zero FermionicSumOp (for `other`)
            if not any(True for _ in other._operator_dict):
                assert self._register_length == other._register_length, \
                    'Operators act on Fermion Registers of different length'
                return other

            # Compute the product (Fermionic type operators consist of a sum of
            # FermionicOperator): F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci
            # are FermionicOperators
            for op1 in self.operator_list:
                for op2 in other.operator_list:
                    new_operatorlist.append(op1 * op2)
            return FermionicSumOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'FermionicSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects or FermionicOperators.
        """
        # Catch the case of a zero FermionicSumOp (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, FermionicOperator):
                assert self._register_length == len(other), \
                    'Operators act on Fermion Registers of different length'
            # return FermionicOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        if isinstance(other, FermionicOperator):
            # Create copy of the FermionicSumOp in which `other` is multiplied by every
            # FermionicOperator
            new_operatorlist = [other * copy.deepcopy(base_operator)
                                for base_operator in self.operator_list]
            return FermionicSumOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'FermionicSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects."""
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'FermionicSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __add__(self, other):
        """Returns a `FermionicSumOp` representing the sum of the given base fermionic
        operators.
        """
        if isinstance(other, FermionicOperator):
            # Create copy of the FermionicSumOp
            new_operatorlist = copy.deepcopy(self.operator_list)

            # Only add the new operator if it has a nonzero-coefficient.
            if not other.coeff == 0:
                # Check compatibility (i.e. operators act on same register length)
                assert self._is_compatible(other), "Incompatible register lengths for '+'. "
                new_operatorlist.append(other)

            return FermionicSumOp(new_operatorlist)

        if isinstance(other, FermionicSumOp):
            new_operatorlist = copy.deepcopy(self.operator_list)
            other_operatorlist = copy.deepcopy(other.operator_list)

            # Check compatibility (i.e. operators act on same register length)
            assert self._is_compatible(other), "Incompatible register lengths for '+'. "

            new_operatorlist += other_operatorlist

            return FermionicSumOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for +: 'FermionicSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a `FermionicSumOp` representing the difference of the given fermionic
        operators.
        """
        if isinstance(other, (FermionicOperator, FermionicSumOp)):
            return self.__add__(-1 * other)

        raise TypeError("Unsupported operand type(s) for -: 'FermionicSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __pow__(self, power):
        """Overloads the power operator `**` for applying an operator `self`, `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")

            if power == 0:
                identity = FermionicSumOp([FermionicOperator('I' * self._register_length)])
                return identity

            operator = copy.deepcopy(self)
            for _ in range(power-1):
                operator *= operator
            return operator

        raise TypeError("Unsupported operand type(s) for **: 'FermionicSumOp' and "
                        "'{}'".format(type(power).__name__))

    @property
    def operator_list(self):
        """Getter for the operator_list of `self`"""
        return list(self._operator_dict.values())

    @property
    def register_length(self):
        """Getter for the length of the fermionic register that the FermionicSumOp `self` acts
        on.
        """
        return self._register_length

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of `self`."""
        daggered_operator_list = [operator.dagger() for operator in self.operator_list]
        return FermionicSumOp(daggered_operator_list)

    def _is_compatible(self, operator) -> bool:
        """
        Checks whether the `operator` is compatible (same shape and

        Args:
            operator (FermionicOperator/FermionicSumOp): a fermionic operator

        Returns:
            True iff `operator` is compatible with `self`.
        """
        same_length = (self.register_length == operator.register_length)
        compatible_type = isinstance(operator, (FermionicOperator, FermionicSumOp))

        if not compatible_type or not same_length:
            return False

        return True

    def to_opflow(self, pauli_table):
        """TODO"""
        ret_op = functools.reduce(lambda x, y: x.add(y), [op.to_opflow(pauli_table)
                                                          for op in self.operator_list])
        return ret_op.reduce()
