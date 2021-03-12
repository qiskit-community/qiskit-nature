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

"""The Super-Fast Bravyi-Kitaev Mapper."""

from qiskit.opflow import PauliSumOp
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp

from .qubit_mapper import QubitMapper


class BravyiKitaevSuperFastMapper(QubitMapper):
    """The Super-Fast Bravyi-Kitaev fermion-to-qubit mapping. """

    def map(self, second_q_op: FermionicOp) -> PauliSumOp:
        # TODO
        raise NotImplementedError()
