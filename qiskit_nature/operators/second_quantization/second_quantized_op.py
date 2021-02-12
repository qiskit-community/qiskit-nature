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

from numbers import Number
from typing import List, Dict

from qiskit_nature import QiskitNatureError
from .fermionic_op import FermionicOp
from .particle_op import ParticleOp
from .spin_op import SpinOp
from .star_algebra import StarAlgebraMixin


class SecondQuantizedOp(StarAlgebraMixin):
    """A general second-quantized operator.

    This class is used to combine operators of different particle type.
    """
    def __init__(self, operator_list: List[ParticleOp]):
        """
        """
        # TODO: validation of operator_list

        self.fermion = None
        self.boson = None
        self.spin: Dict[int, SpinOp] = {}

        for op in operator_list:
            if isinstance(op, FermionicOp) and self.fermion is None:
                self.fermion = op
            elif isinstance(op, FermionicOp) and self.fermion is not None:
                raise QiskitNatureError("Only one FermionicOp can be set in initializer.")
            # if isinstance(op, BosonicOp):
            # if isinstance(op, SpinOp):

    def __repr__(self):
        return f"SecondQuantizedOp([{repr(self.fermion)}])"

    def mul(self, other):
        if not isinstance(other, Number):
            raise TypeError("Unsupported operand type(s) for *: 'SecondQuantizedOperator' and "
                            "'{}'".format(type(other).__name__))

        operator_list = []
        if self.fermion is not None:
            operator_list.append(other * self.fermion)

        return SecondQuantizedOp(operator_list)

    def compose(self, other):
        if not isinstance(other, SecondQuantizedOp):
            raise TypeError("Unsupported operand type(s) for @: 'SecondQuantizedOp' and "
                            "'{}'".format(type(other).__name__))

    def add(self, other):
        if not isinstance(other, SecondQuantizedOp):
            raise TypeError("Unsupported operand type(s) for +: 'SecondQuantizedSumOp' and "
                            "'{}'".format(type(other).__name__))
        # TODO: implement

    def adjoint(self):
        daggered_operator_list = []
        if self.fermion is not None:
            daggered_operator_list.append(self.fermion.dagger)

        return SecondQuantizedOp(daggered_operator_list)
