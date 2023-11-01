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

from qiskit_nature.second_q.operators import MixedOp

from .qubit_mapper import QubitMapper


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
        mappers: Dictionary of mappers corresponding to "local" Hilbert spaces of the global problem.
        hilbert_space_registers: Ordered dictionary of local registers and their respective sizes.
    """

    def __init__(self, mappers: dict[str, QubitMapper], hilbert_space_registers: dict):
        """
        Args:
            mappers: Dictionary of mappers corresponding to the "local" Hilbert spaces.
            hilbert_space_registers: Ordered dictionary of local registers with their sizes.
        """
        super().__init__()
        self.mappers = mappers
        self.hilbert_space_registers = hilbert_space_registers

    def _map_tuple_product(
        self,
        index: tuple[str],
        operator_tuple: tuple[int, ...],
    ):
        """Mapping of operator products.
        When the operator is not present in the tuple, construct a padding SparsePauliOp("II..I")
        """

        coef, op_tuple = operator_tuple[0], operator_tuple[1:]

        tup_dict = {index[k]: self.mappers[index[k]].map(op_tuple[k]) for k in range(len(index))}
        padding_ops = {
            index: SparsePauliOp("I" * value)
            for index, value in self.hilbert_space_registers.items()
        }
        new_dict = {
            index: tup_dict[index] if index in tup_dict else padding_ops[index]
            for index in self.hilbert_space_registers.keys()
        }
        product_op = coef * reduce(SparsePauliOp.tensor, list(new_dict.values())[::-1])

        return product_op.simplify()

    def _distribute_map(self, operator_dict: dict[str, list]):
        """Mapping of operators sums within the various Hilbert spaces."""
        final_op = sum(
            sum(self._map_tuple_product(key, operator_tuple) for operator_tuple in operator_list)
            for key, operator_list in operator_dict.items()
        )
        return final_op

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
