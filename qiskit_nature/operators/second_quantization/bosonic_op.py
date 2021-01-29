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

"""The Bosonic-particle Operator."""
# TODO: currently this is a pure copy-paste of the fermionic_operator.py where "Fermionic" has been
# search-and-replaced with "Bosonic". Thus, the math needs to be updated to correctly reflect
# bosonic properties!

import copy
import numbers

import numpy as np

from .primitives.bosonic_operator import BosonicOperator
from .particle_op import ParticleOp


class BosonicOp(ParticleOp):
    """
    Bosonic type operators

    The abstract fermionic registers are implemented in two subclasses, BosonicOperator and
    BosonicOp, inspired by the implementation of Pauli operators in qiskit. A
    BosonicOperator is the equivalent of a single Pauli string on a qubit register. The
    BosonicOp represents a sum of multiple BosonicOperators. They act on fermionic
    registers of a fixed length determined at the time of initialization.
    """

    def __init__(self, operator_list, register_length=None):

        # 0. Initialize member variables
        if not any(True for _ in operator_list):
            # Treat case of zero operator (empty operator_list)
            assert isinstance(register_length, int), \
                'When instantiating the zero BosonicOp, a register length must be provided.'
            self._register_length = register_length
        else:
            # Treat case of nonzero operator_list
            self._register_length = copy.deepcopy(len(operator_list[0]))

        self._operator_dict = {}

        # Go through all operators in the operator list
        for base_operator in operator_list:
            # 1.  Parse
            # 1.1 Check if they are valid, compatible BosonicOperator instances
            assert isinstance(base_operator, BosonicOperator), \
                'BosonicOperators must be built up from `BosonicOperator` objects'
            assert len(base_operator) == self._register_length, \
                'BosonicOperators must act on fermionic registers of same length.'

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
        # SumOp.__init__(self, particle_type='fermionic')

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
        number-type, a BosonicOperator or a BosonicOp.
        """
        # Catch the case of a zero BosonicOp (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, BosonicOperator):
                assert self._register_length == len(other), \
                    'Operators act on Fermion Registers of different length'
            elif isinstance(other, BosonicOp):
                assert self._register_length == other._register_length, \
                    'Operators act on Fermion Registers of different length'
            # return BosonicOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, (numbers.Number, BosonicOperator)):
            # Create copy of the BosonicOp in which every BosonicOperator is
            # multiplied by `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other
                                for base_operator in self.operator_list]
            return BosonicOp(new_operatorlist)

        if isinstance(other, BosonicOp):
            # Initialize new operator_list for the returned Bosonic operator
            new_operatorlist = []

            # Catch the case of a zero BosonicOp (for `other`)
            if not any(True for _ in other._operator_dict):
                assert self._register_length == other._register_length, \
                    'Operators act on Fermion Registers of different length'
                return other

            # Compute the product (Bosonic type operators consist of a sum of BosonicOperator):
            # F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci are BosonicOperators
            for op1 in self.operator_list:
                for op2 in other.operator_list:
                    new_operatorlist.append(op1 * op2)
            return BosonicOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'BosonicOp' and "
                        "'{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects or BosonicOperators.
        """
        # Catch the case of a zero BosonicOp (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, BosonicOperator):
                assert self._register_length == len(other), \
                    'Operators act on Fermion Registers of different length'
            # return BosonicOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        if isinstance(other, BosonicOperator):
            # Create copy of the BosonicOp in which `other` is multiplied by every
            # BosonicOperator
            new_operatorlist = [other * copy.deepcopy(base_operator)
                                for base_operator in self.operator_list]
            return BosonicOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'BosonicOp' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects."""
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'BosonicOp' and "
                        "'{}'".format(type(other).__name__))

    def __add__(self, other):
        """Returns a `BosonicOp` representing the sum of the given base fermionic
        operators.
        """
        if isinstance(other, BosonicOperator):
            # Create copy of the BosonicOp
            new_operatorlist = copy.deepcopy(self.operator_list)

            # Only add the new operator if it has a nonzero-coefficient.
            if not other.coeff == 0:
                # Check compatibility (i.e. operators act on same register length)
                assert self._is_compatible(other), "Incompatible register lengths for '+'. "
                new_operatorlist.append(other)

            return BosonicOp(new_operatorlist)

        if isinstance(other, BosonicOp):
            new_operatorlist = copy.deepcopy(self.operator_list)
            other_operatorlist = copy.deepcopy(other.operator_list)

            # Check compatibility (i.e. operators act on same register length)
            assert self._is_compatible(other), "Incompatible register lengths for '+'. "

            new_operatorlist += other_operatorlist

            return BosonicOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for +: 'BosonicOp' and "
                        "'{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a `BosonicOp` representing the difference of the given fermionic
        operators.
        """
        if isinstance(other, (BosonicOperator, BosonicOp)):
            return self.__add__(-1 * other)

        raise TypeError("Unsupported operand type(s) for -: 'BosonicOp' and "
                        "'{}'".format(type(other).__name__))

    def __pow__(self, power):
        """Overloads the power operator `**` for applying an operator `self`, `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")

            if power == 0:
                identity = BosonicOp([BosonicOperator('I' * self._register_length)])
                return identity

            operator = copy.deepcopy(self)
            for _ in range(power-1):
                operator *= operator
            return operator

        raise TypeError("Unsupported operand type(s) for **: 'BosonicOp' and "
                        "'{}'".format(type(power).__name__))

    @property
    def operator_list(self):
        """Getter for the operator_list of `self`"""
        return list(self._operator_dict.values())

    @property
    def register_length(self):
        """Getter for the length of the fermionic register that the BosonicOp `self` acts
        on.
        """
        return self._register_length

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of `self`."""
        daggered_operator_list = [operator.dagger() for operator in self.operator_list]
        return BosonicOp(daggered_operator_list)

    def _is_compatible(self, operator) -> bool:
        """
        Checks whether the `operator` is compatible (same shape and

        Args:
            operator (BosonicOperator/BosonicOp): a bosonic operator

        Returns:
            True iff `operator` is compatible with `self`.
        """
        same_length = (self.register_length == operator.register_length)
        compatible_type = isinstance(operator, (BosonicOperator, BosonicOp))

        if not compatible_type or not same_length:
            return False

        return True

    def to_opflow(self, method):
        """TODO"""
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError
