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
import warnings

from typing import List

import numpy as np

from .primitives.bosonic_operator import BosonicSumOp
from .primitives.fermionic_operator import FermionicSumOp
from .primitives.particle_operator import ParticleOperator
from .primitives.spin_operator import SpinSumOp

from .bosonic_sum_op import BosonicSumOp
from .fermionic_sum_op import FermionicSumOp
from .second_quantized_sum_op import SecondQuantizedSumOp
from .spin_sum_op import SpinSumOp


class SecondQuantizedOperator:
    """A class to combine operators that act on different particle type registers. Currently
    supports the following registers: ['fermionic', 'bosonic', 'spin 0.5', 'spin 1.0', etc.]
    """

    def __init__(self, operator_list):
        self._register_operators = {}
        self._registers = []

        # 1. Parse input and fill the member variables
        assert isinstance(operator_list, (list, np.ndarray)), \
            'Please provide a list of operators as input.'

        for operator in operator_list:
            if not isinstance(operator, ParticleOperator):
                raise UserWarning("Elements of `operator_list` must be of `ParticleOperator` type. "
                                  "Allowed operator types include `FermionicSumOp`, "
                                  "`BosonicSumOp`, `SpinSumOp`, and `SpinOperator`.")
            register_name = operator.particle_type
            self[register_name] = operator

    def __getitem__(self, register_name):
        """Getter for the individual operators acting on register of different particle types.

        Args:
            register_name (str): The name of the register on which the operator acts on. Must be one
                of ['fermionic', 'bosonic', 'spin 0.5', 'spin 1.0', etc.]

       Returns:
           ParticleOperator:
               The respective ParticleOperator.
        """
        # Check for correct indexing
        self._is_allowed_key(register_name)

        return self._register_operators[register_name]

    def __setitem__(self, register_name, operator):
        """Setter for the individual operators acting on register of different particle types.

        Args:
            register_name (str): The name of the register on which the operator acts on. Must be one
                                 of ['fermionic', 'bosonic', 'spin 0.5', 'spin 1.0', etc.]
            operator (ParticleOperator): A ParticleOperator object representing the operator on the
                                         specific register

        Raises:
            UserWarning: if an operator mismatches its register.
        """
        # 1. Parse
        #  Check for correct indexing
        self._is_allowed_key(register_name)
        #  Check if the given operator matches the register_name.
        if not register_name == operator.particle_type:
            raise UserWarning("Cannot assign a '{}' type operator to a '{}' register ".format(
                operator.particle_type, register_name))
        # 2. Warn if an operator will be overwritten
        if register_name in self.registers:
            warnings.warn("SecondQuantizedOperator already has a '{}' register. Setting it "
                          "overwrites it.".format(register_name))
        else:
            self._registers.append(register_name)

        # 3. Assign the operator to the respective register
        self._register_operators[register_name] = operator

    def __matmul__(self, other):
        """Implements the operator tensorproduct."""
        if isinstance(other, ParticleOperator):
            new_mixed_operator = copy.deepcopy(self)
            assert other.particle_type not in new_mixed_operator.registers, \
                "Operator already has a '{0}' register. Please include all '{0}' operators " \
                "into this register.".format(other.particle_type)
            new_mixed_operator[other.particle_type] = other

        elif isinstance(other, SecondQuantizedOperator):
            new_mixed_operator = copy.deepcopy(self)
            for register_name in other.registers:
                assert register_name not in new_mixed_operator.registers, \
                    "Operator already has a '{0}' register. Please include all '{0}' operators " \
                    "into this register.".format(register_name)
                new_mixed_operator[register_name] = other[register_name]

        else:
            raise TypeError("Unsupported operand @ for objects of type '{}' and '{}'".format(
                type(self).__name__, type(other).__name__))

        return new_mixed_operator

    def __add__(self, other) -> SecondQuantizedSumOp:
        """Returns a SecondQuantizedSumOp representing the sum of the given operators.
        """
        if isinstance(other, SecondQuantizedOperator):
            is_prop = self.is_proportional_to(other)

            if is_prop[0]:
                return self.__mul__(other=(1+is_prop[1]))

            return SecondQuantizedSumOp([self.copy(), other.copy()])

        if isinstance(other, SecondQuantizedSumOp):
            return other.__add__(self)

        raise TypeError("Unsupported operand type(s) for +: 'SecondQuantizedOperator' and "
                        "'{}'".format(type(other).__name__))

    def __neg__(self):
        """Overload unary -."""
        return self.__mul__(other=-1)

    def __sub__(self, other) -> SecondQuantizedSumOp:
        """Returns a SecondQuantizedSumOp representing the difference to the given
        SecondQuantizedOperator.
        """
        return self.__add__((-1) * other)

    def __mul__(self, other):
        """Overloads the multiplication operator `*` for self and other, where other is a
        number-type.
        """
        if isinstance(other, numbers.Number):
            # Absorb the multiplication factor into the first register (could also be absorbed in
            # any other register)
            first_register_type = self.registers[0]
            new_mixed_operator = copy.deepcopy(self)
            # Catch the warning (from SecondQuantizedOperator.__setitem__(...)) when a register is
            # being updated.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="SecondQuantizedOperator already has a "
                                        "'{}' register. Setting it overwrites "
                                        "it.".format(first_register_type))
                new_mixed_operator[first_register_type] *= other
            return new_mixed_operator

        raise TypeError("Unsupported operand type(s) for *: 'SecondQuantizedOperator' and "
                        "'{}'".format(type(other).__name__))

    def __rmul__(self, other):
        """Overloads the right multiplication operator `*` for multiplication with number-type
        objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        raise TypeError("Unsupported operand type(s) for *: 'SecondQuantizedOperator' and "
                        "'{}'".format(type(other).__name__))

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects.
        """
        if isinstance(other, numbers.Number):
            return self.__mul__(1./other)

        raise TypeError("Unsupported operand type(s) for /: 'SecondQuantizedOperator' and "
                        "'{}'".format(type(other).__name__))

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self."""
        daggered_operator_list = [self[register_type].dagger() for register_type in self.registers]
        return SecondQuantizedOperator(daggered_operator_list)

    def __repr__(self):
        full_str = 'SecondQuantizedOperator acting on registers:'
        for register_name in self.registers:
            full_str += '\n{} :'.ljust(12).format(register_name) + \
                str(self.register_length(register_name))
        return full_str

    def copy(self):
        """Returns a deepcopy of `self`."""
        return copy.deepcopy(self)

    @property
    def registers(self) -> List:
        """Return the particle types that SecondQuantizedOperator `self` acts on. The list order
        corresponds to the order of the tensor product.

        Returns:
            The list of registers of different particle type that the SecondQuantizedOperator acts
            on. This order is also the tensor product order.
        """
        return self._registers

    def print_operators(self):
        """Print the representations of the operators within the SecondQuantizedOperator."""
        full_str = 'SecondQuantizedOperator\n'

        for register in self.registers:
            full_str += (register + ': \n') + self[register].__repr__() + '\n'
        return full_str

    def register_length(self, register_name):
        """Returns the length of the register with name `register_name`."""
        # Check for correct indexing
        self._is_allowed_key(register_name)
        # Return length of the respective register
        return self[register_name].register_length

    @staticmethod
    def _is_allowed_key(key) -> bool:
        """Checks whether `key` is an allowed key, i.e. one of ['fermionic', 'bosonic', 'spin 0.5',
        'spin 1.0', etc.]

        Args:
            key (str): Must be one of ['fermionic', 'bosonic', 'spin 0.5', 'spin 1.0', etc.]

        Returns:
            A boolean whether the key is valid.

        Raises:
            UserWarning: if an disallowed register is encountered.
            TypeError: if an invalid SpinSumOp label is encountered.
        """
        if key == 'fermionic':
            return True

        if key == 'bosonic':
            return True

        if key[0:4] == 'spin':
            spin = float(key[5:])
            if (np.fix(2 * spin) != 2 * spin) or (spin < 0):
                raise TypeError('Spin must be a non-negative integer or half-integer')
            return True

        raise UserWarning("Allowed register arguments are ['fermionic', 'bosonic', 'spin 0.5', "
                          "'spin 1.0', etc.] not '{}'".format(key))

    # pylint: disable=too-many-branches,too-many-locals
    def is_proportional_to(self, other) -> List:
        """Checks whether two SecondQuantizedOperators (M1, M2) are proportional to each other, c *
        M1 = M2, where c is a complex number and M1 = `self` and M2 = `other`. (Used for adding two
        SecondQuantizedOperator type objects)

        Args:
            other (SecondQuantizedOperator): a SecondQuantizedOperator

        Returns:
            Returns a list [bool, numbers.Number] with the corresponding factor of proportionality.
        """
        # Parse for validity and compatibility
        assert isinstance(other, SecondQuantizedOperator), \
            '`other` must be a `SecondQuantizedOperator` type object'
        assert np.array_equal(other.registers, self.registers), \
            'The two SecondQuantizedOperators must act on the same particle type registers in the' \
            ' same order'
        for register_type in self.registers:
            assert other.register_length(register_type) == self.register_length(register_type), \
                "Cannot compare '{}' type operators acting on registers of different " \
                "length".format(register_type)

        # Check for proportionality and calculate the corresponding factor
        factor = 1.  # Define factor of proportionality
        for register_type in self.registers:
            # 0. Convert FermionicOperators to FermionicSumOps, BosonicOperators to
            #    BosonicSumOps and SpinOperators to SpinSumOps
            if isinstance(self[register_type], FermionicOperator):
                register_1 = copy.deepcopy(self[register_type])
                register_1 = FermionicSumOp([register_1])
            elif isinstance(self[register_type], BosonicOperator):
                register_1 = copy.deepcopy(self[register_type])
                register_1 = BosonicSumOp([register_1])
            elif isinstance(self[register_type], SpinOperator):
                register_1 = copy.deepcopy(self[register_type])
                register_1 = SpinSumOp([register_1])
            else:
                register_1 = self[register_type]

            if isinstance(other[register_type], FermionicOperator):
                register_2 = copy.deepcopy(other[register_type])
                register_2 = FermionicSumOp([register_2])
            if isinstance(other[register_type], BosonicOperator):
                register_2 = copy.deepcopy(other[register_type])
                register_2 = BosonicSumOp([register_2])
            elif isinstance(other[register_type], SpinOperator):
                register_2 = copy.deepcopy(other[register_type])
                register_2 = SpinSumOp([register_2])
            else:
                register_2 = other[register_type]

            # 1. Generate dictionaries for the particle type operators
            operator_dict_1 = {}
            operator_dict_2 = {}
            for op1 in register_1.operator_list:
                operator_dict_1[op1.label] = op1.coeff
            for op2 in register_2.operator_list:
                operator_dict_2[op2.label] = op2.coeff

            # 2. Check if all labels of the SecondQuantizedOperators are equal
            label_set_1 = set(operator_dict_1.keys())
            label_set_2 = set(operator_dict_2.keys())
            # 2.1 Check if the two label sets are equal
            if bool(label_set_1.symmetric_difference(label_set_2)):
                return [False, None]

            # 3. Check for proportionality
            # Set a reference label and coefficient and `normalize` the other coefficients according
            # to the reference, in order to compare the operators
            ref_label = list(operator_dict_1.keys())[0]
            ref_coeff_1 = operator_dict_1[ref_label]
            ref_coeff_2 = operator_dict_2[ref_label]
            for op_label in operator_dict_1:
                # Note: iterating a dict, iterates its keys unless specified otherwise
                coeff_1 = operator_dict_1[op_label] / ref_coeff_1
                coeff_2 = operator_dict_2[op_label] / ref_coeff_2
                # if not np.allclose([coeff_1], [coeff_2]):
                if coeff_1 != coeff_2:
                    return [False, None]

            # 4. Update factor of proportionality
            factor *= (ref_coeff_2 / ref_coeff_1)

        return [True, factor]
