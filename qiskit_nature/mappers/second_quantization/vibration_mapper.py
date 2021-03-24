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

"""Vibration Mapper."""

from abc import abstractmethod

from qiskit.opflow import PauliSumOp
from qiskit_nature.operators import VibrationOp

from .qubit_mapper import QubitMapper


class VibrationMapper(QubitMapper):
    """ Mapper of Vibration Operator to Qubit Operator """

    @abstractmethod
    def map(self, second_q_op: VibrationOp) -> PauliSumOp:
        """Maps a class:`VibrationOp` to a `PauliSumOp`.

        Args:
            second_q_op: the :class:`VibrationOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()
