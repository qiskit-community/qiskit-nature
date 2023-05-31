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

"""Spin Mapper."""

from __future__ import annotations

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import SpinOp, MixedOp

from .qubit_mapper import ListOrDictType, QubitMapper


class MixedMapper(QubitMapper):
    """Mapper of a Mixed Operator to a Qubit Operator.

    This class is intended to be used for handling the mapping of composed fermionic and (or) bosonic
    systems, defined as :class:`~qiskit_nature.second_q.operators.MixedOp`, into qubit operators.

    Please note that the creation and usage of this class requires the precise definition of the
    composite Hilbert size corresponding to the problem. It is expected that this "global" Hilbert space
    will result in the tensor product of one or multiple "local" Hilbert spaces, where the ordering of
    the "local" Hilbert spaces must be provided by the user and will correspond to the ordering of the
    corresponding qubit registers before their concatenation.

    The following attributes can be read and updated once the ``TaperedQubitMapper`` object has been
    constructed.

    Attributes:
        mappers: Dictionary of mappers to associate to each "local" Hilbert spaces of the global problem.
    """

    def __init__(
        self,
        mappers: dict[str, QubitMapper],
    ):
        """
        Args:
            mappers: Dictionary of the mappers corresponding to each of the "local" Hilbert spaces.
        """
        super().__init__()
        self.mappers = mappers

    def _map_tuple_product(self, key, operator_tuple, hilbert_space_registers):
        """Mapping of operator products."""
        # initialize with the identity operator in the "global" Hilbert space.
        coefficient = operator_tuple[0]
        dict_mapped_op = {}
        for key_temp, length in hilbert_space_registers.items():
            dict_mapped_op[key_temp] = SparsePauliOp("I" * length, coeffs=coefficient)

        # for each Hilbert space appearing in the product of operators, replace the identity with the
        # corresponding term.
        for index, op in enumerate(operator_tuple[1:]):
            key_char = key[index]
            mapper = self.mappers[key_char]
            dict_mapped_op[key_char] = mapper.map(op)  # TODO: compose()

        # tensor all the elements of the dictionary.
        list_op = list(dict_mapped_op.values())
        product_op = coefficient * list_op[0]
        for op in list_op[1:]:
            product_op = product_op.tensor(op)

        return product_op  # TODO: simplify()

    def _map_list_sum(self, key, operator_list, hilbert_space_registers):
        """Mapping of operators sums within same Hilbert spaces."""
        final_op = sum(
            self._map_tuple_product(key, operator_tuple, hilbert_space_registers)
            for operator_tuple in operator_list
        )
        return final_op

    def _map_dict_sum(self, operator_dict, hilbert_space_registers):
        """Mapping of operators sums within various Hilbert spaces."""
        final_op = sum(
            self._map_list_sum(key, operator_tuple, hilbert_space_registers)
            for key, operator_tuple in operator_dict.data.items()
        )
        return final_op

    def map(
        self,
        mixed_op: MixedOp,
        *,
        hilbert_space_registers: dict[str, int],
    ) -> SparsePauliOp:
        """Map the :class:`~qiskit_nature.second_q.operators.MixedOp` into a qubit operator.

        The ``MixedOp`` is a representation of sums of products of operators corresponding to different Hilbert spaces. The mapping procedure first runs through all of the terms to be summed, and then maps the operator product corresponding to each summand by tensoring the individually mapped operators.

        Args:
            hilbert_space_registers: Ordered dictionary attributing a register length to each of the
                Hilbert space. Note that the ordering of the "local" Hilbert spaces in the dictionary
                translates directly to the ordering of the corresponding qubit registers.
        """

        mapped_op = self._map_dict_sum(mixed_op, hilbert_space_registers)
        return mapped_op
