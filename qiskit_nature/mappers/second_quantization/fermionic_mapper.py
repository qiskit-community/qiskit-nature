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

"""Fermionic Mapper."""

from abc import abstractmethod

from qiskit.opflow import PauliSumOp
from qiskit_nature.operators.second_quantization import FermionicOp

from .qubit_mapper import QubitMapper


class FermionicMapper(QubitMapper):
    """Mapper of Fermionic Operator to Qubit Operator"""

    @abstractmethod
    def map(self, second_q_op: FermionicOp) -> PauliSumOp:
        """Maps a class:`FermionicOp` to a `PauliSumOp`.

        Args:
            second_q_op: the :class:`FermionicOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()
