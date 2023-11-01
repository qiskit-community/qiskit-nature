# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A QubitMapper wrapper to transform from block-ordered to interleaved qubits."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import FermionicOp

from .fermionic_mapper import FermionicMapper


class InterleavedQubitMapper(FermionicMapper):
    """A ``FermionicMapper`` wrapper returning interleaved-ordered operators.

    This class is intended to be used with fermionic systems. Furthermore, it is designed to work
    with ``FermionicMapper`` classes which map fermionic operators to the qubit space by site
    (unlike for example the :class:`~qiskit_nature.second_q.mappers.BravyiKitaevSuperFastMapper`
    which maps by interactions).

    .. warning::

       The mapper will _not_ perform any assertions on the wrapped ``FermionicMapper``. Thus,
       wrapping a :class:`~qiskit_nature.second_q.mappers.BravyiKitaevSuperFastMapper` is valid code
       which will indeed produce qubit operators for you. You will just not be able to interpret the
       order of the qubits in the same way.

    .. warning::

        The builtin two-qubit reduction of the :class:`.ParityMapper` will also not provide correct
        results when combined with this mapper. Again, this is not asserted so be aware of this
        pitfall.
        Thus, if you would like to reduce the number of qubits, you should instead look towards the
        :class:`.TaperedQubitMapper` which removes qubits based on all Z2-symmetries it detects in
        the operator.

    For site-based mappers, Qiskit Nature always arranges the qubits corresponding to the alpha-spin
    and beta-spin components in a blocked fashion. I.e. the first half of the qubit register
    corresponds to the alpha-spin components, and the second half to the beta-spin one, like so:

    .. code-block::

       a1, a2, ..., aN, b1, b2, ..., bN

    This class allows you to wrap such a ``FermionicMapper`` to produce qubit operators which have
    an interleaved order of qubits, instead. Taking the example from before, the outcome will be the
    following:

    .. code-block::

       a1, b1, a2, b2, ..., aN, bN

    .. note::

       This reordering is intended for an even total number of spin orbitals (i.e. the alpha-spin
       and beta-spin components should be identical in length; which they usually are). However,
       this is not asserted, so reordering a qubit operator label of odd length will still happen.

    Here is a very simple usage example:

    .. code-block:: python

       from qiskit_nature.second_q.mappers import JordanWignerMapper, InterleavedQubitMapper
       from qiskit_nature.second_q.operators import FermionicOp

       blocked_mapper = JordanWignerMapper()
       interleaved_mapper = InterleavedQubitMapper(blocked_mapper)

       ferm_op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=4)

       blocked_op = blocked_mapper.map(ferm_op)
       # SparsePauliOp(['IIXY', 'IIYY', 'IIXX', 'IIYX'], coeffs=[-0.25j, 0.25, 0.25, 0.25j])

       print(interleaved_mapper.map(ferm_op))
       # SparsePauliOp(['IXIY', 'IYIY', 'IXIX', 'IYIX'], coeffs=[-0.25j, 0.25, 0.25, 0.25j])

    The following attributes can be set via the initializer but can also be read and updated once
    the ``InterleavedQubitMapper`` object has been constructed.

    Attributes:
        mapper (FermionicMapper): the actual mapper for mapping from :class:`.FermionicOp` to qubit
            operators.
    """

    def __init__(self, mapper: FermionicMapper):
        """
        Args:
            mapper: the actual ``FermionicMapper`` mapping :class:`.FermionicOp` to qubit operators.
        """
        self.mapper = mapper

    def _map_single(
        self, second_q_op: FermionicOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        if register_length is None:
            register_length = second_q_op.register_length

        interleaved_sec_op = second_q_op.permute_indices(
            list(range(0, register_length, 2)) + list(range(1, register_length, 2))
        )

        interleaved_op = self.mapper._map_single(
            interleaved_sec_op, register_length=register_length
        )

        return interleaved_op
