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

"""The Sum Operator base interface."""

import copy
from abc import ABC, abstractmethod
from typing import Any, List


class SumOp(ABC):
    """The Sum Operator base interface.

    This interface should be implemented by all creation- and annihilation-type particle operators
    in the second-quantized formulation.
    """

    def __init__(self, particle_type):
        self._particle_type = particle_type

    @property
    def particle_type(self):
        """Return the particle type"""
        return copy.deepcopy(self._particle_type)

    @property
    @abstractmethod
    def register_length(self) -> int:
        """Getter for the length of the particle register that the SumOp acts on."""
        raise NotImplementedError

    @property
    @abstractmethod
    def operator_list(self) -> List[Any]:
        """Getter for the operator_list of the `SumOp`"""
        raise NotImplementedError

    @abstractmethod
    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        raise NotImplementedError

    @abstractmethod
    def to_opflow(self, pauli_table):
        """TODO"""
        raise NotImplementedError
