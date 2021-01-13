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

"""The Second-Quantized Operator."""

import copy
import numbers

import numpy as np

from .second_quantized_operator import SecondQuantizedOperator


class SecondQuantizedSumOp:
    """A general SecondQuantizedSumOp.

    This class represents sums of mixed operators, i.e. linear combinations of
    SecondQuantizedOperators with identical particle type registers.
    """

    def __init__(self, mixed_operator_list):

        self._registers = mixed_operator_list[0].registers
        self._register_lengths = {}
        for register_type in self._registers:
            self._register_lengths[register_type] = \
                mixed_operator_list[0].register_length(register_type)

        # Check if the elements of the mixed_operator_list are valid & compatible instances of the
        # SecondQuantizedOperator class
        for mixed_operator in mixed_operator_list:
            assert isinstance(mixed_operator, SecondQuantizedOperator), \
                'SecondQuantizedSumOp must be built up from `SecondQuantizedOperator` objects'
            assert np.array_equal(mixed_operator.registers, self._registers), \
                'SecondQuantizedSumOp elements must act on the same particle type registers in' \
                ' the same order'
            for register_type in self._registers:
                assert mixed_operator.register_length(register_type) == \
                    self._register_lengths[register_type], "Cannot sum '{}' type operators acting" \
                    " on registers of different length".format(register_type)

        # TODO: Find a way to 'factorize' the operator, such that each element only appears once in
        # the operator_list
        self._operator_list = mixed_operator_list

    @property
    def operator_list(self):
        """Returns the operator list."""
        return self._operator_list

    @property
    def registers(self):
        """Returns the register list."""
        return self._registers

    def register_length(self, register_type):
        """Returns the length of the register with name `register_name`."""
        assert register_type in self.registers, "The SecondQuantizedOperatpr does not contain a " \
            "register of type '{}'".format(register_type)
        return self._register_lengths[register_type]

    def __repr__(self):
        full_str = 'SecondQuantizedSumOp acting on registers:'
        for register_name in self.registers:
            full_str += '\n{} :'.ljust(12).format(register_name) + \
                str(self.register_length(register_name))
        full_str += '\nTotal number of SecondQuantizedOperators: {}'.format(len(self.operator_list))
        return full_str

    def __add__(self, other):
        """Returns a SecondQuantizedSumOp representing the sum of the given operators.
        """
        if isinstance(other, SecondQuantizedOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            # If the operators are proportional to each other, simply update coefficients
            for idx, operator in enumerate(new_operatorlist):
                is_prop = operator.is_proportional_to(other)
                if is_prop[0]:
                    operator *= (1+is_prop[1])
                    new_operatorlist[idx] = operator
                    return SecondQuantizedSumOp(new_operatorlist)
            # Else, just append the new operator to the operator_list
            new_operatorlist.append(other)
            return SecondQuantizedSumOp(new_operatorlist)

        if isinstance(other, SecondQuantizedSumOp):
            new_operatorlist = copy.deepcopy(self.operator_list)
            for elem in other.operator_list:
                new_operatorlist.append(elem)
                # If the operators are proportional to each other, simply update coefficients
                for idx, operator in enumerate(new_operatorlist[:-1]):
                    is_prop = operator.is_proportional_to(elem)
                    if is_prop[0]:
                        new_operatorlist.pop()
                        operator *= (1 + is_prop[1])
                        new_operatorlist[idx] = operator
                        break
                # Else, the new operator has been added to the operator_list
            return SecondQuantizedSumOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for +: 'SecondQuantizedSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def __sub__(self, other):
        """Returns a SecondQuantizedSumOp representing the difference of the given
        SecondQuantizedOperators.
        """
        return self.__add__((-1) * other)

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type.
        """
        if isinstance(other, numbers.Number):
            # Create copy of the SpinOperator in which every BaseSpinOperator is multiplied by
            # `other`.
            new_operatorlist = [copy.deepcopy(mixed_operator) * other
                                for mixed_operator in self.operator_list]
            return SecondQuantizedSumOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'SecondQuantizedSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        raise TypeError("Unsupported operand type(s) for *: 'SecondQuantizedSumOp' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'SecondQuantizedSumOp' and "
                        "'{}'".format(type(other).__name__))

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self."""
        daggered_operator_list = [mixed_operator.dagger() for mixed_operator in self.operator_list]
        return SecondQuantizedSumOp(daggered_operator_list)

    def copy(self):
        """Returns a deepcopy of `self`."""
        return copy.deepcopy(self)

    def print_operators(self):
        """Print the representations of the operators within the SecondQuantizedOperator."""
        full_str = 'SecondQuantizedSumOp\n'

        for operator in self.operator_list:
            full_str += operator.print_operators() + '\n'
        return full_str
