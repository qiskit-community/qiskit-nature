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
import itertools
import numbers

from typing import Union

import numpy as np

from .particle_operator import ParticleOperator


class BosonicOperator(ParticleOperator):
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
            return BosonicOperator(label=self.label, coeff=other * self.coeff)

        raise TypeError("Unsupported operand type(s) for *: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type or a BosonicOperator.
        """
        if isinstance(other, numbers.Number):
            return BosonicOperator(label=self.label, coeff=other * self.coeff)

        if isinstance(other, BosonicOperator):
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
                    return BosonicOperator('I'*len(self), coeff=0.)
                new_label += new_char

            return BosonicOperator(new_label, new_coeff)

        # Multiplication with a BosonicSumOp is implemented in the __rmul__ method of the
        # BosonicSumOp class
        from ..bosonic_sum_op import BosonicSumOp
        if isinstance(other, BosonicSumOp):
            # TODO: this should probably be something like:
            # return other.__rmul__(self)
            raise NotImplementedError

        raise TypeError("Unsupported operand type(s) for *: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __pow__(self, power):
        """Overloads the power operator `**` for applying an operator `self` `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")

            if power == 0:
                identity = BosonicOperator('I' * len(self))
                return identity

            operator = copy.deepcopy(self)
            for _ in range(power-1):
                operator *= operator
            return operator

        raise TypeError("Unsupported operand type(s) for **: 'BosonicOperator' and "
                        "'{}'".format(type(power).__name__))

    def __add__(self, other) -> Union['BosonicSumOp', 'BosonicOperator']:  # type: ignore
        """Returns a fermionic operator representing the sum of the given BosonicOperators"""
        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..bosonic_sum_op import BosonicSumOp

        if isinstance(other, BosonicOperator):
            # Case 1: `other` is a `BosonicOperator`.
            #  In this case we add the two operators, if they have non-zero coefficients. Otherwise
            #  we simply return the operator that has a non-vanishing coefficient (self, if both
            #  vanish).
            if other.coeff == 0:
                return copy.deepcopy(self)

            if self.coeff == 0:
                return copy.deepcopy(other)

            if self.label == other.label:
                return BosonicOperator(self.label, self.coeff + other.coeff)

            return BosonicSumOp([copy.deepcopy(self), copy.deepcopy(other)])

        if isinstance(other, BosonicSumOp):
            # Case 2: `other` is a `BosonicSumOp`.
            #  In this case use the __add__ method of BosonicSumOp.
            return other.__add__(self)

        # Case 3: `other` is any other type. In this case we raise an error.
        raise TypeError("Unsupported operand type(s) for +: 'BosonicOperator' and "
                        "'{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a fermionic operator representing the sum of the given BosonicOperators."""

        if not isinstance(other, BosonicOperator):
            raise TypeError("Unsupported operand type(s) for -: 'BosonicOperator' and "
                            "'{}'".format(type(other).__name__))

        return self.__add__(-1 * other)

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: `BosonicOperator` and "
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
            normal_ordered_operator_list.append(BosonicOperator(label=label,
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

        return BosonicOperator(label=daggered_label, coeff=phase * np.conj(self.coeff))

    def to_opflow(self, pauli_table):
        """TODO"""
        raise NotImplementedError
