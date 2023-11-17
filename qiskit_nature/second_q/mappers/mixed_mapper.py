# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Mixed Mapper class."""

from __future__ import annotations
from functools import reduce

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import MixedOp, SparseLabelOp

from .qubit_mapper import QubitMapper


class MixedMapper(QubitMapper):
    """Mapper of a Mixed Operator to a Qubit Operator.

    This class is intended to be used for handling the mapping of composed fermionic and (or) bosonic
    systems, defined as :class:`~qiskit_nature.second_q.operators.MixedOp`, into qubit operators.

    Please note that the creation and usage of this class requires the precise definition of the
    composite Hilbert size corresponding to the problem.
    The ordering of the qubit registers associated to the bosonic and fermionic degrees of freedom (for example)
    must be provided by the user through the definition of the hilbert space registers dictionary. This
    ordering corresponds to a specific way to take the tensor product of the fermionic and bosonic operators.

    .. note::

      This class is limited to one instance of a Fermionic Hilbert space, to ensure the anticommutation
      relations of the fermions.

    .. note::

      This class enforces the register lengths to the mappers. Note that for the bosonic mappers, the
      register lengths is not directly equal to the qubit register length but to the number of modes.
      See the documentation of the class :class:``~BosonicLinearMapper``.

    The following attributes can be read and updated once the ``MixedMapper`` object has been constructed.

    Attributes:
        mappers: Dictionary of mappers corresponding to "local" Hilbert spaces of the global problem.
        hilbert_space_register_lengths: Ordered dictionary of local registers and their respective sizes.
    """

    def __init__(self, mappers: dict[str, QubitMapper], hilbert_space_register_lengths: dict):
        """
        Args:
            mappers: Dictionary of mappers corresponding to the "local" Hilbert spaces.
            hilbert_space_register_lengths: Ordered dictionary of local registers with their sizes.
        """
        super().__init__()
        self.mappers = mappers
        self.hilbert_space_register_lengths = hilbert_space_register_lengths

    def _map_tuple_product(
        self, active_indices: tuple[str], active_operators: tuple[SparseLabelOp]
    ) -> SparsePauliOp:
        """Maps a product of operators defined on the local Hilbert spaces defined by the active
        indices. Note that the order of the active indices is not relevant. The only relevant ordering
        is that given by :attr:`~MixedMapper.hilbert_space_register_lengths` at initialization.

        When the operator is not present in the tuple, we use a padding operator with identities.

        Args:
            active_indices: Specificiation of the Hilbert spaces on which the operator acts.
            active_operators: List of operators to compose..
        """

        product_op_dict = {
            index: SparsePauliOp("I" * value)
            for index, value in self.hilbert_space_register_lengths.items()
        }

        for active_index, active_op in zip(active_indices, active_operators):
            register_length = self.hilbert_space_register_lengths[active_index]
            product_op_dict[active_index] = self.mappers[active_index].map(
                active_op, register_length=register_length
            )

        product_op = reduce(SparsePauliOp.tensor, list(product_op_dict.values()))

        return product_op

    def _distribute_map(self, operator_dict: dict[str, list]) -> SparsePauliOp:
        """Distributes the mapping of operators to each of the terms defined across specific
        Hilbert spaces.

        Args:
            operator_dict: Dictionary of (key, operator list) pairs where the key specify which
                local "Hilbert space" the operators act on. Note that the first element of the
                operator list is the coefficient of this operator product across these Hilbert spaces.
        """

        mapped_op: SparsePauliOp = 0
        for active_indices, operator_list in operator_dict.items():
            for coef_and_operators in operator_list:
                coef, active_operators = coef_and_operators[0], coef_and_operators[1:]
                mapped_op += coef * self._map_tuple_product(active_indices, active_operators)

        return mapped_op.simplify()

    def map(
        self,
        mixed_op: MixedOp,
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp:
        """Map the :class:`~qiskit_nature.second_q.operators.MixedOp` into a qubit operator.

        The ``MixedOp`` is a representation of sums of products of operators corresponding to different
        Hilbert spaces. The mapping procedure first runs through all of the terms to be summed,
        and then maps the operator product corresponding to each summand by tensoring the
        individually mapped operators.

        Args:
            mixed_op: Operator to map.
            register_length: UNUSED.
        """

        mapped_op = self._distribute_map(mixed_op.data)

        return mapped_op
