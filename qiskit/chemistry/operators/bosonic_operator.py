# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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
import itertools
import numbers

from typing import Union

import numpy as np

from .particle_operator import ParticleOperator


class BosonicOperator(ParticleOperator):
    """
    Bosonic type operators

    The abstract fermionic registers are implemented in two subclasses, BaseBosonicOperator and
    BosonicOperator, inspired by the implementation of Pauli operators in qiskit. A
    BaseBosonicOperator is the equivalent of a single Pauli string on a qubit register. The
    BosonicOperator represents a sum of multiple BaseBosonicOperators. They act on fermionic
    registers of a fixed length determined at the time of initialization.
    """

    def __init__(self, operator_list, register_length=None):

        # 0. Initialize member variables
        if not any(True for _ in operator_list):
            # Treat case of zero operator (empty operator_list)
            assert isinstance(register_length, int), \
                'When instantiating the zero BosonicOperator, a register length must be provided.'
            self._register_length = register_length
        else:
            # Treat case of nonzero operator_list
            self._register_length = copy.deepcopy(len(operator_list[0]))

        self._operator_dict = {}

        # Go through all operators in the operator list
        for base_operator in operator_list:
            # 1.  Parse
            # 1.1 Check if they are valid, compatible BaseBosonicOperator instances
            assert isinstance(base_operator, BaseBosonicOperator), \
                'BosonicOperators must be built up from `BaseBosonicOperator` objects'
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
        ParticleOperator.__init__(self, particle_type='fermionic')

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
        number-type, a BaseBosonicOperator or a BosonicOperator.
        """
        # Catch the case of a zero BosonicOperator (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, BaseBosonicOperator):
                assert self._register_length == len(other), \
                    'Operators act on Fermion Registers of different length'
            elif isinstance(other, BosonicOperator):
                assert self._register_length == other._register_length, \
                    'Operators act on Fermion Registers of different length'
            # return BaseBosonicOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, (numbers.Number, BaseBosonicOperator)):
            # Create copy of the BosonicOperator in which every BaseBosonicOperator is
            # multiplied by `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other
                                for base_operator in self.operator_list]
            return BosonicOperator(new_operatorlist)

        if isinstance(other, BosonicOperator):
            # Initialize new operator_list for the returned Bosonic operator
            new_operatorlist = []

            # Catch the case of a zero BosonicOperator (for `other`)
            if not any(True for _ in other._operator_dict):
                assert self._register_length == other._register_length, \
                    'Operators act on Fermion Registers of different length'
                return other

            # Compute the product (Bosonic type operators consist of a sum of BaseBosonicOperator):
            # F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci are BaseBosonicOperators
            for op1 in self.operator_list:
                for op2 in other.operator_list:
                    new_operatorlist.append(op1 * op2)
            return BosonicOperator(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects or BaseBosonicOperators.
        """
        # Catch the case of a zero BosonicOperator (for `self`)
        if not any(True for _ in self._operator_dict):
            if isinstance(other, BaseBosonicOperator):
                assert self._register_length == len(other), \
                    'Operators act on Fermion Registers of different length'
            # return BaseBosonicOperator('I'*self._register_length, coeff = 0.)
            return self

        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        if isinstance(other, BaseBosonicOperator):
            # Create copy of the BosonicOperator in which `other` is multiplied by every
            # BaseBosonicOperator
            new_operatorlist = [other * copy.deepcopy(base_operator)
                                for base_operator in self.operator_list]
            return BosonicOperator(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects."""
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __add__(self, other):
        """Returns a `BosonicOperator` representing the sum of the given base fermionic
        operators.
        """
        if isinstance(other, BaseBosonicOperator):
            # Create copy of the BosonicOperator
            new_operatorlist = copy.deepcopy(self.operator_list)

            # Only add the new operator if it has a nonzero-coefficient.
            if not other.coeff == 0:
                # Check compatibility (i.e. operators act on same register length)
                assert self._is_compatible(other), "Incompatible register lengths for '+'. "
                new_operatorlist.append(other)

            return BosonicOperator(new_operatorlist)

        if isinstance(other, BosonicOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            other_operatorlist = copy.deepcopy(other.operator_list)

            # Check compatibility (i.e. operators act on same register length)
            assert self._is_compatible(other), "Incompatible register lengths for '+'. "

            new_operatorlist += other_operatorlist

            return BosonicOperator(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for +: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a `BosonicOperator` representing the difference of the given fermionic
        operators.
        """
        if isinstance(other, (BaseBosonicOperator, BosonicOperator)):
            return self.__add__(-1 * other)

        raise TypeError("Unsupported operand type(s) for -: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __pow__(self, power):
        """Overloads the power operator `**` for applying an operator `self`, `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")

            if power == 0:
                identity = BosonicOperator([BaseBosonicOperator('I' * self._register_length)])
                return identity

            operator = copy.deepcopy(self)
            for _ in range(power-1):
                operator *= operator
            return operator

        raise TypeError("Unsupported operand type(s) for **: 'BosonicOperator' and "
                        "'{}'".format(type(power).__name__))

    @property
    def operator_list(self):
        """Getter for the operator_list of `self`"""
        return list(self._operator_dict.values())

    @property
    def register_length(self):
        """Getter for the length of the fermionic register that the BosonicOperator `self` acts
        on.
        """
        return self._register_length

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of `self`."""
        daggered_operator_list = [operator.dagger() for operator in self.operator_list]
        return BosonicOperator(daggered_operator_list)

    def _is_compatible(self, operator) -> bool:
        """
        Checks whether the `operator` is compatible (same shape and

        Args:
            operator (BaseBosonicOperator/BosonicOperator): a bosonic operator

        Returns:
            True iff `operator` is compatible with `self`.
        """
        same_length = (self.register_length == operator.register_length)
        compatible_type = isinstance(operator, (BaseBosonicOperator, BosonicOperator))

        if not compatible_type or not same_length:
            return False

        return True


class BaseBosonicOperator(ParticleOperator):
    """A class for simple products (not sums) of fermionic operators on several fermionic modes."""

    def __init__(self, label, coeff=1.):
        # 1. Parse input
        # Parse coeff
        if not isinstance(coeff, numbers.Number):
            raise TypeError("`coeff` must be a number type not '{}'".format(type(coeff).__name__))

        # Parse label
        for char in label:
            # I encodes the identity
            # + the creation operator in the fermion mode at the given position
            # - the annihilation operator in the fermion mode at the given position
            # N denotes the occupation number operator
            # E denotes 1-N (the `emptiness number` operator)
            assert char in ['I', '+', '-', 'N', 'E'], "Label must be a string consisting only of " \
                "['I','+','-','N','E'] not: {}".format(char)

        # 2. Initialize member variables
        self.coeff = coeff
        self.label = label

        # 3. Set the particle type
        ParticleOperator.__init__(self, particle_type='fermionic')

    def __len__(self) -> int:
        """Returns the number of of fermion modes in the fermionic register, i.e. the length of
        `self.label`.
        """
        return len(self.label)

    def __repr__(self) -> str:
        # 1. Treat the case of the zero operator
        if self.coeff == 0:
            return 'zero operator ({})'.format(len(self.label))

        # 2. Treat the general case
        return '{1} \t {0}'.format(self.coeff, self.label)

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects.
        """
        if isinstance(other, numbers.Number):
            return BaseBosonicOperator(label=self.label, coeff=other * self.coeff)

        raise TypeError("Unsupported operand type(s) for *: 'BaseBosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type or a BaseBosonicOperator.
        """
        if isinstance(other, numbers.Number):
            return BaseBosonicOperator(label=self.label, coeff=other * self.coeff)

        if isinstance(other, BaseBosonicOperator):
            assert len(self) == len(other), "Operators act on Fermion Registers of different length"

            new_coeff = self.coeff * other.coeff
            new_label = ''

            # Map the products of two operators on a single fermionic mode to their result.
            mapping = {
                # 0                   - if product vanishes,
                # new label           - if product does not vanish
                'II': 'I',
                'I+': '+',
                'I-': '-',
                'IN': 'N',
                'IE': 'E',

                '+I': '+',
                '++': 0,
                '+-': 'N',
                '+N': 0,
                '+E': '+',

                '-I': '-',
                '-+': 'E',
                '--': 0,
                '-N': '-',
                '-E': 0,

                'NI': 'N',
                'N+': '+',
                'N-': 0,
                'NN': 'N',
                'NE': 0,

                'EI': 'E',
                'E+': 0,
                'E-': '-',
                'EN': 0,
                'EE': 'E'
            }

            for i, char1, char2 in zip(np.arange(len(self)), self.label, other.label):
                # if char2 is one of `-`, `+` we pick up a phase when commuting it to the position
                # of char1
                if char2 in ['-', '+']:
                    # Construct the string through which we have to commute
                    permuting_through = self.label[i + 1:]
                    # Count the number of times we pick up a minus sign when commuting
                    ncommutations = permuting_through.count('+') + permuting_through.count('-')
                    new_coeff *= (-1) ** ncommutations

                # Check what happens to the symbol
                new_char = mapping[char1 + char2]
                if new_char == 0:
                    return BaseBosonicOperator('I'*len(self), coeff=0.)
                new_label += new_char

            return BaseBosonicOperator(new_label, new_coeff)

        # Multiplication with a BosonicOperator is implemented in the __rmul__ method of the
        # BosonicOperator class
        if isinstance(other, BosonicOperator):
            raise NotImplementedError

        raise TypeError("Unsupported operand type(s) for *: 'BaseBosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __pow__(self, power):
        """Overloads the power operator `**` for applying an operator `self` `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")

            if power == 0:
                identity = BaseBosonicOperator('I' * len(self))
                return identity

            operator = copy.deepcopy(self)
            for _ in range(power-1):
                operator *= operator
            return operator

        raise TypeError("Unsupported operand type(s) for **: 'BaseBosonicOperator' and "
                        "'{}'".format(type(power).__name__))

    def __add__(self, other) -> Union[BosonicOperator, 'BaseBosonicOperator']:
        """Returns a fermionic operator representing the sum of the given BaseBosonicOperators"""

        if isinstance(other, BaseBosonicOperator):
            # Case 1: `other` is a `BaseBosonicOperator`.
            #  In this case we add the two operators, if they have non-zero coefficients. Otherwise
            #  we simply return the operator that has a non-vanishing coefficient (self, if both
            #  vanish).
            if other.coeff == 0:
                return copy.deepcopy(self)

            if self.coeff == 0:
                return copy.deepcopy(other)

            if self.label == other.label:
                return BaseBosonicOperator(self.label, self.coeff + other.coeff)

            return BosonicOperator([copy.deepcopy(self), copy.deepcopy(other)])

        if isinstance(other, BosonicOperator):
            # Case 2: `other` is a `BosonicOperator`.
            #  In this case use the __add__ method of BosonicOperator.
            return other.__add__(self)

        # Case 3: `other` is any other type. In this case we raise an error.
        raise TypeError("Unsupported operand type(s) for +: 'BaseBosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a fermionic operator representing the sum of the given BaseBosonicOperators."""

        if not isinstance(other, BaseBosonicOperator):
            raise TypeError("Unsupported operand type(s) for -: 'BaseBosonicOperator' and "
                            "'{}'".format(type(other).__name__))

        return self.__add__(-1 * other)

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: `BaseBosonicOperator` and "
                        "'{}'".format(type(other).__name__))

    @property
    def register_length(self):
        return len(self.label)

    @property
    def operator_list(self):
        return [self]

    def is_normal(self) -> bool:
        """Returns True iff `self.label` is normal ordered.

        Returns:
            True iff the product `self.label` is normal ordered (i.e. - on each mode to the right, +
            to left).
        """
        return self.label.count('E') == 0

    def normal_order(self) -> list:
        """Returns the list of normal-order components of `self`."""
        # Catch the case of the zero-operator:
        if self.coeff == 0:
            return []

        # Set up an empty list in which to save the normal ordered expansion
        normal_ordered_operator_list = []

        # Split the `self.label` at every non-normal ordered symbol (only E = -+)
        splits = self.label.split('E')
        # Count the number of splits
        nsplits = self.label.count('E')

        # Generate all combinations of (I,N) of length nsplits
        combos = list(map(''.join, itertools.product('IN', repeat=nsplits)))

        for combo in combos:
            # compute the sign of the given label combination
            sign = (-1) ** combo.count('N')

            # build up the label token
            label = splits[0]
            for link, next_base in zip(combo, splits[1:]):
                label += link + next_base
            # append the current normal ordered part to the list of the normal ordered expansion
            normal_ordered_operator_list.append(BaseBosonicOperator(label=label,
                                                                    coeff=sign * self.coeff))

        return normal_ordered_operator_list

    def dagger(self):
        """Returns the adjoint (dagger) of `self`."""
        daggered_label = ''

        dagger_map = {
            '+': '-',
            '-': '+',
            'I': 'I',
            'N': 'N',
            'E': 'E'
        }

        phase = 1.
        for i, char in enumerate(self.label):
            daggered_label += dagger_map[char]
            if char in ['+', '-']:
                permute_through = self.label[i+1:]
                phase *= (-1) ** (permute_through.count('+') + permute_through.count('-'))

        return BaseBosonicOperator(label=daggered_label, coeff=phase * np.conj(self.coeff))
