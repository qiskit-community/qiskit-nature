# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
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

import logging
from abc import ABC
from functools import reduce

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit_nature.second_q.operators import MixedOp, SparseLabelOp, FermionicOp

from .qubit_mapper import QubitMapper, _ListOrDict

LOGGER = logging.getLogger(__name__)


class MixedMapper(ABC):
    """Mapper of a Mixed Operator to a Qubit Operator.

    This class is to be used to map systems with particles of different nature,
    such as bosons and fermions.

    Please note that the creation and usage of this class requires the precise definition of the
    composite Hilbert size corresponding to the problem.
    The ordering of the qubit registers associated to the different physical particles
    must be provided by the user through the definition of the Hilbert space registers dictionary.

    This ordering corresponds to a specific way to take the tensor product of the operators.

    .. code-block:: python

        # Consider the Hilbert spaces of a fermionic system Hf and a bosonic system Hb
        bos_mapper = BosonicLinearMapper(max_occupation=1)
        fer_mapper = JordanWignerMapper()
        # These mappers map to qubit simulation Hilbert spaces S(Hf) and S(Hb)
        mappers = {"b1": bos_mapper, "f1": fer_mapper}
        # This ordering of the dictionary implies that the simulation Hilbert spaces are
        # tensored as S(Hf).tensor(S(Hb))
        # This follows the qiskit convention for stacking qubit registers from right to left.
        hilbert_space_register_lengths = {"b1": 1, "f1": 1}
        hilbert_space_register_types = {"b1": BosonicOp, "f1": FermionicOp}
        # One bosonic mode (yet unknown local dimension d)
        # One fermionic mode (qubit).
        mix_mapper = MixedMapper(
            mappers=mappers,
            hilbert_space_register_lengths=hilbert_space_register_lengths,
            hilbert_space_register_types=hilbert_space_register_types,
        )
        # The final simulation register is composed of d+1 qubits arranged as [qf_0, qb_d, ... qb_0]


    .. note::

      This class is limited to one instance of a Fermionic Hilbert space, to ensure the anticommutation
      relations of the fermions.

    .. note::

      This class leaves the handling of register lengths to the mappers.
      Note that for the bosonic mappers, the register lengths is not directly equal to the qubit
      register length but to the number of modes.
      See the documentation of the class :class:``~.BosonicLinearMapper``.

    The following attributes can be read and updated once the ``MixedMapper`` object has been
    constructed.

    Attributes:
        mappers: Dictionary of mappers corresponding to "local" Hilbert spaces of the global problem.
        hilbert_space_register_lengths: Ordered dictionary of local registers sizes.
        hilbert_space_register_types: Ordered dictionary of local registers types.
    """

    def __init__(
        self,
        mappers: dict[str, QubitMapper],
        hilbert_space_register_lengths: dict[str, int],
        hilbert_space_register_types: dict[str, type[SparseLabelOp]],
    ):
        """
        Args:
            mappers: Dictionary of mappers corresponding to the "local" Hilbert spaces.
            hilbert_space_register_lengths: Ordered dictionary of local registers sizes.
            hilbert_space_register_types: Ordered dictionary of local register types.
        """
        super().__init__()
        self.mappers: dict[str, QubitMapper] = mappers
        self.hilbert_space_register_lengths: dict[str, int] = hilbert_space_register_lengths
        self.hilbert_space_register_types: dict[str, type[SparseLabelOp]] = (
            hilbert_space_register_types
        )

        # Only one fermionic register allowed to ensure fermionic statistics.
        count_fermionic_registers = [
            issubclass(register_type, FermionicOp)
            for register_type in self.hilbert_space_register_types.values()
        ]
        if sum(count_fermionic_registers) > 1:
            raise ValueError("Register types can only contain a single fermionic register")

    def _map_tuple_product(
        self, active_indices: tuple[str], active_operators: tuple[SparseLabelOp]
    ) -> SparsePauliOp:
        """Maps a product of operators defined on the local Hilbert spaces defined by the active
        indices. Note that the order of the active indices is not relevant. The only relevant ordering
        is that given by :attr:`~MixedMapper.hilbert_space_register_lengths` at initialization.

        When the operator is not present in the tuple, we use a padding operator with identities.

        Args:
            active_indices: Reference names of the Hilbert spaces on which the operator acts.
            active_operators: List of operators to compose..
        """

        product_op_dict = {
            index: self.mappers[index].map(
                self.hilbert_space_register_types[index].one(), register_length=value
            )
            for index, value in self.hilbert_space_register_lengths.items()
        }

        for active_index, active_op in zip(active_indices, active_operators):

            product_op_dict[active_index] = self.mappers[active_index].map(active_op)
            # Cannot use register_length here because it does not correspond to the
            # qubit register length for all mappers.

        product_op = reduce(SparsePauliOp.tensor, list(product_op_dict.values())[::-1])

        return product_op

    def _distribute_map(self, operator_dict: dict[tuple[str], list[tuple]]) -> SparsePauliOp:
        """Distributes the mapping of operators to each of the terms defined across specific
        Hilbert spaces.

        Args:
            operator_dict: Dictionary of (key, operator list) pairs where the key specify which
                local "Hilbert space" the operators act on. Note that the first element of the
                operator list is the coefficient of this operator product across these Hilbert spaces.
        """

        mapped_op: SparsePauliOp = 0.0
        for active_indices, operator_list in operator_dict.items():
            for coef_and_operators in operator_list:
                coef = coef_and_operators[0]
                active_operators = coef_and_operators[1:]
                mapped_op += coef * self._map_tuple_product(active_indices, active_operators)

        return mapped_op.simplify()

    def _map_single(
        self,
        mixed_op: MixedOp,
    ) -> SparsePauliOp:
        """Maps the :class:`~qiskit_nature.second_q.operators.MixedOp` into a qubit operator.

        The mapping procedure first runs through all of the terms to be summed,
        and then maps the operator product by tensoring the individually mapped operators.

        Args:
            mixed_op: Operator to map.
            register_length: Ignored. The register lengths must be set in the individual mappers.
        """

        mapped_op: SparsePauliOp = self._distribute_map(mixed_op.data)

        return mapped_op

    def map(
        self,
        mixed_ops: MixedOp | ListOrDictType[MixedOp],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp | ListOrDictType[SparsePauliOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            mixed_ops: A second quantized operator, or list thereof.
            register_length: Ignored. The register lengths must be set in the individual mappers.

        Returns:
            A qubit operator in the form of a ``SparsePauliOp``, or list (resp. dict) thereof if a
            list (resp. dict) of second quantized operators was supplied.
        """

        if register_length is not None:
            LOGGER.warning("Argument register length = %s was ignored.", register_length)

        wrapped_second_q_ops, wrapped_type = _ListOrDict.wrap(mixed_ops)

        qubit_ops: _ListOrDict = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = self._map_single(second_q_op)

        returned_ops = qubit_ops.unwrap(wrapped_type)
        # Note the output of the mapping will never be None for standard mappers other than the
        # TaperedQubitMapper.
        return returned_ops
