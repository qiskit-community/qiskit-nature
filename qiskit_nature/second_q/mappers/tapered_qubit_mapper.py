# This code is part of Qiskit.
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

"""The Tapered Qubit Mapper."""

from __future__ import annotations

import logging
from typing import cast

from qiskit.algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries
from qiskit.quantum_info.operators import SparsePauliOp

from qiskit_nature import settings
from qiskit_nature.second_q.operators import SparseLabelOp

from .qubit_mapper import QubitMapper, _ListOrDict

logger = logging.getLogger(__name__)


class TaperedQubitMapper(QubitMapper):
    """The wrapper around qubit mappers implementing the logic to reduce the size of a problem (operator)
    based on mathematical ``Z2Symmetries`` that can be automatically detected in the operator.

    The following attributes can be read and updated once the ``TaperedQubitMapper`` object has been
    constructed.

    Attributes:
        mapper: Object defining the mapping of second quantized operators to Pauli operators.
        z2symmetries: Symmetries to use to reduce the Pauli operators.
    """

    def __init__(
        self,
        mapper: QubitMapper,
        z2symmetries: Z2Symmetries = Z2Symmetries([], [], []),
    ):
        """
        Args:
            mapper: ``QubitMapper`` object implementing the mapping of second quantized operators to
                Pauli operators.
            z2symmetries: ``Z2Symmetries`` object defining the symmetries to use to reduce the Pauli
                operators.

        Raises:
            ValueError: If the input mapper is already a ``TaperedQubitMapper``.
        """
        super().__init__()
        if isinstance(mapper, TaperedQubitMapper):
            raise ValueError(
                "TaperedQubitMapper cannot be nested in another TaperedQubitMapper. "
                "If you want to update your TaperedQubitMapper instance please "
                "build a new one starting from the standard mappers."
            )
        self.mapper: QubitMapper = mapper
        self.z2symmetries = z2symmetries

    def _map_clifford_single(
        self, second_q_op: SparseLabelOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        mapped_op = self.mapper.map(second_q_op, register_length=register_length)
        if isinstance(mapped_op, PauliSumOp):
            mapped_op = mapped_op.primitive
        converted_op = self.z2symmetries.convert_clifford(mapped_op)
        return converted_op

    def _taper_clifford_single(self, converted_op: SparsePauliOp) -> SparsePauliOp:
        # Mappers do not apply symmetry reduction if the tapering values were not set to specify the
        # eigen-sector in which lies the solution.
        if self.z2symmetries.tapering_values is None:
            return converted_op
        else:
            tapered_op = cast(SparsePauliOp, self.z2symmetries.taper_clifford(converted_op))
            return tapered_op

    def _map_single(
        self, second_q_op: SparseLabelOp, *, register_length: int | None = None
    ) -> SparsePauliOp | PauliSumOp:
        converted_op = self._map_clifford_single(second_q_op, register_length=register_length)
        tapered_op = self._taper_clifford_single(converted_op)
        return PauliSumOp(tapered_op) if settings.use_pauli_sum_op else tapered_op

    def map_clifford(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp | PauliSumOp | ListOrDictType[SparsePauliOp | PauliSumOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the internal mapper. Then, composes all mapped pauli operators with the clifford operations
        defined by the internal ``Z2Symmetries`` to prepare for the symmetry reduction.
        This composition gives isospectral operators and exposes redundant qubits for later tapering.

        Args:
            second_q_ops: A second quantized operator, or list (resp. dict) thereof.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the operator being mapped. This is possible because the
            ``register_length`` is considered a lower bound in a ``SparseLabelOp``.

        Returns:
            A qubit operator in the form of a ``PauliSumOp`` or ``SparsePauliOp`` (depending on
            :attr:`~qiskit_nature.settings.use_pauli_sum_op`), or list (resp. dict) thereof if a
            list (resp. dict) of second quantized operators was supplied.
        """
        wrapped_second_q_ops, wrapped_type = _ListOrDict.wrap(second_q_ops)

        qubit_ops: _ListOrDict[SparsePauliOp] = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = self._map_clifford_single(
                second_q_op, register_length=register_length
            )

        # NOTE: _ListOrDict.unwrap takes care of the conversion to/from PauliSumOp based on
        # settings.use_pauli_sum_op
        returned_ops = qubit_ops.unwrap(wrapped_type)

        return returned_ops

    def taper_clifford(
        self,
        pauli_ops: SparsePauliOp | PauliSumOp | ListOrDictType[SparsePauliOp | PauliSumOp],
        *,
        check_commutes: bool = True,
        suppress_none: bool = True,
    ) -> SparsePauliOp | PauliSumOp | None | ListOrDictType[SparsePauliOp | PauliSumOp | None]:
        """Applies the symmetry reduction on a ``PauliSumOp`` or a list (resp. dict). This method implies
        that the second quantized operators were already mapped to Pauli operators and composed with the
        clifford operations defined in the symmetry, for example using the ``map_clifford`` method.

        Args:
            pauli_ops: A pauli operator already evolved with the symmetry clifford operations.
            check_commutes: If the commutativity of operators with symmetries must be checked before
                any calculation.
            suppress_none: If None should be placed in the output list where an operator
                did not commute with symmetry, to maintain order, or whether that should
                be suppressed where the output list length may then be smaller than the input.

        Returns:
            A qubit operator in the form of a ``PauliSumOp`` or ``SparsePauliOp`` (depending on
            :attr:`~qiskit_nature.settings.use_pauli_sum_op`), or list (resp. dict) thereof if a
            list (resp. dict) of second quantized operators was supplied.
        """
        wrapped_pauli_ops, wrapped_type = _ListOrDict.wrap(pauli_ops)

        qubit_ops: _ListOrDict[SparsePauliOp | PauliSumOp]
        if self.z2symmetries.is_empty():
            qubit_ops = wrapped_pauli_ops
        else:
            qubit_ops = _ListOrDict()
            for name, pauli_op in iter(wrapped_pauli_ops):
                if isinstance(pauli_op, PauliSumOp):
                    pauli_op = pauli_op.primitive
                if check_commutes and not self._check_commutes(pauli_op):
                    qubit_ops[name] = None
                else:
                    qubit_ops[name] = self._taper_clifford_single(pauli_op)

        # NOTE: _ListOrDict.unwrap takes care of the conversion to/from PauliSumOp based on
        # settings.use_pauli_sum_op
        returned_ops: SparsePauliOp | PauliSumOp | ListOrDictType[
            SparsePauliOp | PauliSumOp
        ] = qubit_ops.unwrap(wrapped_type, suppress_none=suppress_none)

        return returned_ops

    def map(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp | PauliSumOp | None | ListOrDictType[SparsePauliOp | PauliSumOp | None]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the ``SparseLabelOp`` being mapped. This is possible because the
                ``register_length`` is considered a lower bound in a ``SparseLabelOp``.

        Returns:
            A qubit operator in the form of a ``PauliSumOp`` or ``SparsePauliOp`` (depending on
            :attr:`~qiskit_nature.settings.use_pauli_sum_op`), or list (resp. dict) thereof if a
            list (resp. dict) of second quantized operators was supplied.
        """
        # NOTE: we do not rely on the `_map_single` method here because we want to ensure that
        # `check_commutes=True` is set. This is not done via `_map_single` because the correct
        # filtering of `None` values can only be done on the level of `taper_clifford`
        pauli_ops = self.map_clifford(second_q_ops, register_length=register_length)
        # This choice of keyword arguments ensures that the output does not contain None.
        tapered_ops = self.taper_clifford(pauli_ops, check_commutes=True, suppress_none=True)
        return tapered_ops

    def _check_commutes(self, qubit_op: SparsePauliOp) -> bool:
        logger.debug("Checking operator commutes with symmetries:")
        # We use sq_paulis instead of symmetries because the qubit operator was already composed with the
        # cliffords defined in the symmetry.
        converted_symmetries = self.z2symmetries._sq_paulis
        commuting_rows = qubit_op.paulis.commutes_with_all(converted_symmetries)
        commutes = len(commuting_rows) == qubit_op.size
        logger.debug("  '%s' commutes: %s", id(qubit_op), commutes)
        return commutes
