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
from typing import cast, TYPE_CHECKING

from qiskit.algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries
from qiskit.quantum_info.operators import SparsePauliOp, Pauli

from qiskit_nature.second_q.operators import SparseLabelOp

from .qubit_mapper import QubitMapper, _ListOrDict

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

logger = logging.getLogger(__name__)


class TaperedQubitMapper(QubitMapper):  # pylint: disable=missing-class-docstring
    """The wrapper around qubit mappers implementing the logic to reduce the size of a problem (operator)
    based on mathematical `Z2Symmetries` that can be automatically detected in the operator.

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
        """
        super().__init__()
        self.mapper: QubitMapper = mapper
        self.z2symmetries = z2symmetries

    @classmethod
    def from_problem(
        cls, mapper: QubitMapper, problem: "ElectronicStructureProblem"
    ) -> "TaperedQubitMapper":
        # implies the previous z2symmetries == "auto" case
        # extract problem.hamiltonian and problem.symmetry_sector_locator
        """Builds a ``TaperedQubitMapper`` from one of the other mappers and from a specific problem.
        This simplifies the identification of the Pauli operator symmetries and of the symmetry sector
        in which lies the solution of the problem.

        Args:
            mapper: ``QubitMapper`` object implementing the mapping of second quantized operators to
                Pauli operators.
            problem: A class encoding a problem to be solved.

        Return:
            A ``TaperedQubitMapper`` with pre-built symmetry specifications.
        """

        qubit_op, _ = problem.second_q_ops()
        mapped_op = mapper.map(qubit_op).primitive
        z2_symmetries = Z2Symmetries.find_z2_symmetries(mapped_op)
        tapering_values = problem.symmetry_sector_locator(z2_symmetries, mapper)
        z2_symmetries.tapering_values = tapering_values
        return TaperedQubitMapper(mapper, z2_symmetries)

    def _map_clifford_single(self, second_q_op: SparseLabelOp) -> SparsePauliOp:
        mapped_op = self.mapper.map(second_q_op).primitive
        converted_op = self.z2symmetries.convert_clifford(mapped_op)
        return converted_op

    def _symmetry_reduce_clifford_single(
        self, converted_op: SparsePauliOp, check_commutes: bool = True
    ) -> None | SparsePauliOp:

        if self.z2symmetries.is_empty() or self.z2symmetries.tapering_values is None:
            return converted_op
        elif check_commutes:
            logger.debug("Checking operator commute with symmetries:")
            converted_symmetries = self.z2symmetries._sq_paulis
            commutes = TaperedQubitMapper._check_commutes(converted_symmetries, converted_op)

            if not commutes:
                return None

        tapered_op = self.z2symmetries.taper_clifford(converted_op)
        cast(SparsePauliOp, tapered_op)
        return tapered_op

    def _map_single(self, second_q_op: SparseLabelOp) -> PauliSumOp | None:
        converted_op = self._map_clifford_single(second_q_op)
        tapered_op = self._symmetry_reduce_clifford_single(converted_op)

        returned_op = PauliSumOp(tapered_op) if isinstance(tapered_op, SparsePauliOp) else None
        return returned_op

    def map_clifford(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
    ) -> PauliSumOp | ListOrDictType[PauliSumOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            second_q_ops: A second quantized operator, or list (resp. dict) thereof.

        Returns:
            A qubit operator in the form of a PauliSumOp, or list (resp. dict) thereof if a list
            (resp. dict) of second quantized operators was supplied.
        """
        wrapped_type = type(second_q_ops)

        if issubclass(wrapped_type, SparseLabelOp):
            second_q_ops = [second_q_ops]

        wrapped_second_q_ops: _ListOrDict[SparseLabelOp] = _ListOrDict(second_q_ops)

        qubit_ops: _ListOrDict = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = PauliSumOp(self._map_clifford_single(second_q_op))

        returned_ops: PauliSumOp | ListOrDictType[PauliSumOp]
        returned_ops = qubit_ops.unwrap(wrapped_type)

        return returned_ops

    def symmetry_reduce_clifford(
        self,
        pauli_ops: PauliSumOp | ListOrDictType[PauliSumOp],
        check_commutes: bool = True,
        suppress_none: bool = False,
    ) -> PauliSumOp | ListOrDictType[PauliSumOp]:
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
            A qubit operator in the form of a PauliSumOp, or list (resp. dict) thereof if a list
            (resp. dict) of second quantized operators was supplied.
        """
        wrapped_type = type(pauli_ops)

        if issubclass(wrapped_type, PauliSumOp):
            pauli_ops = [pauli_ops]
            suppress_none = False

        wrapped_pauli_ops: _ListOrDict[PauliSumOp] = _ListOrDict(pauli_ops)

        qubit_ops: _ListOrDict[PauliSumOp] = _ListOrDict()
        for name, pauli_op in iter(wrapped_pauli_ops):
            qubit_op = self._symmetry_reduce_clifford_single(pauli_op.primitive, check_commutes)
            qubit_ops[name] = PauliSumOp(qubit_op) if qubit_op is not None else qubit_op

        returned_ops: PauliSumOp | ListOrDictType[PauliSumOp] = qubit_ops.unwrap(
            wrapped_type, suppress_none=suppress_none
        )

        return returned_ops

    @staticmethod
    def _check_commutes(sq_paulis: list[Pauli], qubit_op: SparsePauliOp) -> bool:
        # commutes = []
        commuting_rows = qubit_op.paulis.commutes_with_all(sq_paulis)
        commutes = len(commuting_rows) == qubit_op.size
        logger.debug("  '%s' commutes: %s", id(qubit_op), commutes)
        return commutes
