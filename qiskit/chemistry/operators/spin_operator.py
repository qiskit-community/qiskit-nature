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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the `FermionicOperator` and `BosonicOperator`
as it relies an the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

import copy
import numbers

import numpy as np

from .particle_operator import ParticleOperator


class SpinOperator(ParticleOperator):
    """
    Spin type operators. This class represents sums of `spin strings`, i.e. linear combinations of
    BaseSpinOperators with same spin and register length.
    """

    def __init__(self, operator_list):
        # 1. Parse input
        self._register_length = len(operator_list[0])
        self.spin = operator_list[0].spin
        for elem in operator_list:
            assert isinstance(elem, BaseSpinOperator)
            assert len(elem) == self._register_length, \
                'Cannot sum operators acting on registers of different length'
            assert elem.spin == self.spin, \
                'Cannot sum operators with different spins.'

        # 2. Initialize the operator list of `self`
        self._operator_list = operator_list

        # 3. Set the operators particle type to 'spin S' with S the spin value (as float with 1
        # decimal).
        ParticleOperator.__init__(self, particle_type='spin {0:.1f}'.format(self.spin))

    @property
    def register_length(self):
        return self._register_length

    def __repr__(self):
        full_str = ''
        for operator in self._operator_list:
            full_str += '{1} \t {0}\n'.format(operator.coeff, operator.label)
        return full_str

    # TODO: Make this much more efficient by working with lists and label indices
    def __add__(self, other):
        """Returns a SpinOperator representing the sum of the given operators.
        """
        if isinstance(other, BaseSpinOperator):
            new_operatorlist = copy.deepcopy(self.operator_list)
            # If the operators have the same label, simply add coefficients:
            for operator in new_operatorlist:
                if other.label == operator.label:
                    sum_coeff = operator.coeff + other.coeff
                    operator._coeff = sum_coeff  # set the coeff of sum operator to sum_coeff
                    # if the new coefficient is zero, remove the operator from the list
                    if sum_coeff == 0:
                        new_operatorlist.remove(operator)
                    return SpinOperator(new_operatorlist)
            new_operatorlist.append(other)
            return SpinOperator(new_operatorlist)

        if isinstance(other, SpinOperator):
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
            return SpinOperator(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for +: 'SpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def __sub__(self, other):
        """Returns a SpinOperator representing the difference of the given BaseSpinOperators.
        """
        return self.__add__((-1) * other)

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type.
        """
        if isinstance(other, numbers.Number):
            # Create copy of the SpinOperator in which every BaseSpinOperator is multiplied by
            # `other`.
            new_operatorlist = [copy.deepcopy(base_operator) * other
                                for base_operator in self.operator_list]
            return SpinOperator(new_operatorlist)

        raise TypeError("Unsupported operand type(s) for *: 'SpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        raise TypeError("Unsupported operand type(s) for *: 'SpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'SpinOperator' and "
                        "'{}'".format(type(other).__name__))

    @property
    def operator_list(self):
        return self._operator_list

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self."""
        daggered_operator_list = [operator.dagger() for operator in self.operator_list]
        return SpinOperator(daggered_operator_list)


class BaseSpinOperator(ParticleOperator):
    """A class for products and powers of XYZ-ordered Spin operators."""

    # pylint: disable=too-many-arguments
    def __init__(self, spin, spin_x, spin_y, spin_z, coeff=1.):
        """TODO."""

        # 1. Infer the number of individual spin systems in the register
        self._register_length = len(spin_x)

        # 2. Parse input
        # Parse spin
        if (np.fix(2 * spin) != 2 * spin) or (spin < 0):
            raise TypeError('spin must be a non-negative integer or half-integer')

        # Parse coeff
        if not isinstance(coeff, numbers.Number):
            raise TypeError("`coeff` must be a number type not '{}'".format(type(coeff).__name__))

        for spin_operators in [spin_x, spin_y, spin_z]:
            # Check the input type (arrays)
            if not isinstance(spin_operators, (list, np.ndarray)):
                raise TypeError("spin_x, spin_y and spin_z must be `np.ndarray` with integers, not "
                                "'{}'".format(type(spin_operators).__name__))

            # Check the length
            assert len(spin_operators) == self._register_length, \
                "`spin_x, spin_y, spin_z` must be of same length."

            # Check datatype of first elements
            if not isinstance(spin_operators[0], (int, np.integer)):
                raise TypeError("Elements of `spin_x, spin_y, spin_z` must be of integer type.")

        # 3. Initialize the member variables
        self._spin = spin
        self._coeff = coeff
        self._spin_x = np.asarray(spin_x).astype(dtype=np.uint16, copy=True)
        self._spin_y = np.asarray(spin_y).astype(dtype=np.uint16, copy=True)
        self._spin_z = np.asarray(spin_z).astype(dtype=np.uint16, copy=True)
        self._label = None
        self.generate_label()

        # 4. Set the operators particle type to 'spin S' with S the spin value (as float with 1
        # decimal).
        ParticleOperator.__init__(self, particle_type='spin {0:.1f}'.format(self.spin))

    @property
    def spin(self):
        """The spin value of the individual spin systems in the register. The dimension of the
        spin systems is therefore 2S+1."""
        return self._spin

    @property
    def coeff(self):
        """The (complex) coefficient of the spin operator."""
        return self._coeff

    @property
    def spin_x(self):
        """A np.ndarray storing the power i of (spin) X operators on the spin system.
        I.e. [0, 4, 2] corresponds to X0^0 \\otimes X1^4 \\otimes X2^2, where Xi acts on the i-th
        spin system in the register.
        """
        return self._spin_x

    @property
    def spin_y(self):
        """A np.ndarray storing the power i of (spin) Y operators on the spin system.
        I.e. [0, 4, 2] corresponds to Y0^0 \\otimes Y1^4 \\otimes Y2^2, where Yi acts on the i-th
        spin system in the register.
        """
        return self._spin_y

    @property
    def spin_z(self):
        """A np.ndarray storing the power i of (spin) Z operators on the spin system.
        I.e. [0, 4, 2] corresponds to Z0^0 \\otimes Z1^4 \\otimes Z2^2, where Zi acts on the i-th
        spin system in the register.
        """
        return self._spin_z

    @property
    def register_length(self):
        return self._register_length

    @property
    def label(self):
        """The description of `self` in terms of a string label."""
        return self._label

    @property
    def operator_list(self):
        return [self]

    def __len__(self) -> int:
        """Returns the number of spin systems in the spin register, i.e. the length of `self.spin_x`
        (or spin_y, spin_z).
        """
        return self._register_length

    def __repr__(self) -> str:
        """Prints `self.coeff` and `self.label` to the console."""
        return self.label + ' \t ' + str(self.coeff)

    def __eq__(self, other):
        """Overload == ."""
        if not isinstance(other, BaseSpinOperator):
            return False

        spin_equals = (self.spin == other.spin)
        spin_x_equals = np.all(self.spin_x == other.spin_x)
        spin_y_equals = np.all(self.spin_y == other.spin_y)
        spin_z_equals = np.all(self.spin_z == other.spin_z)
        coeff_equals = np.all(self.coeff == other.coeff)

        return spin_equals and spin_x_equals and spin_y_equals and spin_z_equals and coeff_equals

    def __ne__(self, other):
        """Overload != ."""
        return not self.__eq__(other)

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def generate_label(self):
        """Generates the string description of `self`."""
        label = ''
        for pos, n_x, n_y, n_z in zip(np.arange(self._register_length),
                                      self.spin_x, self.spin_y, self.spin_z):
            if n_x > 0:
                label += ' X^{}'.format(n_x)
            if n_y > 0:
                label += ' Y^{}'.format(n_y)
            if n_z > 0:
                label += ' Z^{}'.format(n_z)
            if n_x > 0 or n_y > 0 or n_z > 0:
                label += '[{}] |'.format(pos)
            else:
                label += ' I[{}] |'.format(pos)

        # remove leading and trailing whitespaces and trailing |
        self._label = label[1:-2]
        return self.label

    def __add__(self, other):
        """Returns a SpinOperator representing the sum of the given BaseSpinOperators.
        """

        if isinstance(other, BaseSpinOperator):
            # If the operators have the same label, simply add coefficients:
            if other.label == self.label:
                sum_coeff = self.coeff + other.coeff
                # create a copy of the initial operator to preserve initialize transformations
                sum_operator = copy.deepcopy(self)
                # set the coeff of sum operator to sum_coeff
                sum_operator._coeff = sum_coeff
                return sum_operator
            return SpinOperator([copy.deepcopy(self), copy.deepcopy(other)])

        if isinstance(other, SpinOperator):
            #  In this case use the __add__ method of FermionicOperator.
            return other.__add__(self)

        raise TypeError("Unsupported operand type(s) for +: 'BaseSpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def __sub__(self, other):
        """Returns a SpinOperator representing the difference of the given BaseSpinOperators.
        """
        return self.__add__((-1) * other)

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        raise TypeError("Unsupported operand type(s) for *: 'BaseSpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type object.
        """
        if isinstance(other, numbers.Number):
            # create a copy of self (to also preserve pre-computed transforms)
            product_operator = copy.deepcopy(self)
            product_operator._coeff *= other
            return product_operator

        raise TypeError("Unsupported operand type(s) for *: 'BaseSpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'BaseSpinOperator' and "
                        "'{}'".format(type(other).__name__))

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        # Note: X, Y, Z are hermitian, therefore the dagger operation on a BaseSpinOperator amounts
        # to simply complex conjugating the coefficient.
        # create a copy of self (to also preserve pre-computed transforms)
        new_operator = copy.deepcopy(self)
        # pylint: disable=protected-access
        new_operator._coeff = np.conj(self.coeff)
        return new_operator
