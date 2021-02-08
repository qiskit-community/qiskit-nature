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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the `FermionicOperator` and `BosonicOperator`
as it relies an the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

from typing import Optional
import copy
import numbers

from .primitives.spin_operator import SpinOperator
from .particle_op import ParticleOp


class SpinOp(ParticleOp):
    """
    Spin type operators. This class represents sums of `spin strings`, i.e. linear combinations of
    SpinOperators with same spin and register length.
    """

    def __init__(self, operator_list):
        # 1. Parse input
        self._register_length = len(operator_list[0])
        self.spin = operator_list[0].spin
        for elem in operator_list:
            assert isinstance(elem, SpinOperator)
            assert len(elem) == self._register_length, \
                'Cannot sum operators acting on registers of different length'
            assert elem.spin == self.spin, \
                'Cannot sum operators with different spins.'

        # 2. Initialize the operator list of `self`
        self._operator_list = operator_list

        # 3. Set the operators particle type to 'spin S' with S the spin value (as float with 1
        # decimal).
        # SumOp.__init__(self, particle_type='spin {0:.1f}'.format(self.spin))

    @property
    def register_length(self):
        return self._register_length

    def __repr__(self):
        full_str = ''
        for operator in self._operator_list:
            full_str += '{1} \t {0}\n'.format(operator.coeff, operator.label)
        return full_str

    # TODO: Make this much more efficient by working with lists and label indices
    def add(self, other):
        """Returns a SpinOp representing the sum of the given operators.
        """
        if isinstance(other, SpinOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            # If the operators have the same label, simply add coefficients:
            for operator in new_operatorlist:
                if other.label == operator.label:
                    sum_coeff = operator.coeff + other.coeff
                    operator._coeff = sum_coeff  # set the coeff of sum operator to sum_coeff
                    # if the new coefficient is zero, remove the operator from the list
                    if sum_coeff == 0:
                        new_operatorlist.remove(operator)
                    return SpinOp(new_operatorlist)
            new_operatorlist.append(other)
            return SpinOp(new_operatorlist)

        if isinstance(other, SpinOp):
            new_operatorlist = copy.deepcopy(self.operator_list)
            for elem in other.operator_list:
                new_operatorlist.append(elem)
                # If the operators have the same label, simply add coefficients:
                for operator in new_operatorlist[:-1]:
                    if elem.label == operator.label:
                        new_operatorlist.pop()
                        sum_coeff = operator.coeff + elem.coeff
                        operator._coeff = sum_coeff  # set the coeff of sum operator to sum_coeff
                        # if the new coefficient is zero, remove the operator from the list
                        if sum_coeff == 0:
                            new_operatorlist.remove(operator)
                        break
            return SpinOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for +: 'SpinOp' and "
                        "'{}'".format(type(other).__name__))

    def compose(self, other):
        raise NotImplementedError()

    def mul(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type.
        """
        if isinstance(other, numbers.Number):
            # Create copy of the SpinOp in which every SpinOperator is multiplied by
            # `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other
                                for base_operator in self.operator_list]
            return SpinOp(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'SpinOp' and "
                        "'{}'".format(type(other).__name__))

    @property
    def operator_list(self):
        """operator list"""
        return self._operator_list

    def adjoint(self):
        """Returns the complex conjugate transpose (dagger) of self."""
        daggered_operator_list = [operator.dagger() for operator in self.operator_list]
        return SpinOp(daggered_operator_list)

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None):
        raise NotImplementedError
