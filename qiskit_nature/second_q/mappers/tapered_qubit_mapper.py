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
from copy import deepcopy
from typing import TYPE_CHECKING

from qiskit.algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries
from qiskit.quantum_info.operators import SparsePauliOp, PauliList

from qiskit_nature.second_q.operators import SparseLabelOp

from .parity_mapper import ParityMapper
from .qubit_mapper import QubitMapper, _ListOrDict

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

logger = logging.getLogger(__name__)


class TaperedQubitMapper(QubitMapper):  # pylint: disable=missing-class-docstring
    """The wrapper around qubit mappers implementing the logic for Z2 symmetry reduction to reduce a
    problem (operator) size based on mathematical symmetries that can be detected in the operator.
    """

    def __init__(
        self,
        mapper: QubitMapper,
        z2symmetries: Z2Symmetries = Z2Symmetries([], [], []),
    ):
        """

        Args:
            mapper (QubitMapper): _description_
            z2symmetries (list[int]): _description_
        """
        super().__init__()
        self._mapper: QubitMapper = deepcopy(mapper)
        self.z2symmetries = z2symmetries
        self.check_commutes: bool = True

    @property
    def num_particles(self) -> tuple[int, int] | None:
        """Gets the number of particles."""
        if isinstance(self._mapper, ParityMapper):
            return self._mapper.num_particles
        else:
            return None

    @num_particles.setter
    def num_particles(self, value: tuple[int, int] | None) -> None:
        """Sets the numbers of particle."""
        if isinstance(self._mapper, ParityMapper):
            self._mapper.num_particles = value

    @classmethod
    def from_problem(
        cls, mapper: QubitMapper, problem: "ElectronicStructureProblem"
    ) -> "TaperedQubitMapper":
        # implies the previous z2symmetries == "auto" case
        # extract problem.hamiltonian and problem.symmetry_sector_locator
        """_summary_

        Args:
            mapper (QubitMapper): _description_
            problem (ElectronicStructureProblem): _description_

        Return:
            Tapered Qubit Mapper
        """

        qubit_op, _ = problem.second_q_ops()
        mapped_op = mapper.map(qubit_op).primitive
        z2_symmetries = Z2Symmetries.find_z2_symmetries(mapped_op)
        tapering_values = problem.symmetry_sector_locator(z2_symmetries, mapper)
        z2_symmetries.tapering_values = tapering_values
        return TaperedQubitMapper(mapper, z2_symmetries)

    def _map_clifford_single(self, second_q_op: SparseLabelOp) -> SparsePauliOp:
        mapped_op = self._mapper.map(second_q_op).primitive
        converted_op = self.z2symmetries.convert_clifford(mapped_op)
        return converted_op

    def _symmetry_reduce_clifford_single(
        self, converted_op: SparsePauliOp
    ) -> None | SparsePauliOp | list[SparsePauliOp]:
        """_summary_

        Args:
            converted_op (SparsePauliOp): _description_

        Returns:
            SparsePauliOp | list[SparsePauliOp]: _description_
        """
        if self.z2symmetries.is_empty():
            tapered_op = converted_op
        elif self.check_commutes:
            logger.debug("Checking operators commute with symmetry:")

            commutes = TaperedQubitMapper._check_commutes(
                self.z2symmetries._sq_paulis, converted_op
            )

            if commutes:
                tapered_op = self.z2symmetries.taper_clifford(converted_op)
            else:
                tapered_op = None
        else:
            tapered_op = self.z2symmetries.taper_clifford(converted_op)

        return tapered_op

    def _map_single(self, second_q_op: SparseLabelOp) -> PauliSumOp | list[PauliSumOp]:
        converted_op = self._map_clifford_single(second_q_op)
        tapered_op = self._symmetry_reduce_clifford_single(converted_op)

        returned_op: PauliSumOp | list[PauliSumOp] | None

        if tapered_op is None:
            return None
        elif isinstance(tapered_op, SparsePauliOp):
            returned_op = PauliSumOp(tapered_op)
        else:
            returned_op = [PauliSumOp(op) for op in tapered_op]
        return returned_op

    def map_clifford(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
        suppress_none: bool = None,
    ) -> PauliSumOp | ListOrDictType[PauliSumOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof.
            suppress_none: If None should be placed in the output list where an operator
                did not commute with symmetry, to maintain order, or whether that should
                be suppressed where the output list length may then be smaller than the input.

        Returns:
            A qubit operator in the form of a PauliSumOp, or list (resp. dict) thereof if a list
            (resp. dict) of second quantized operators was supplied.
        """
        wrapped_type = type(second_q_ops)

        if issubclass(wrapped_type, SparseLabelOp):
            second_q_ops = [second_q_ops]
            suppress_none = False

        wrapped_second_q_ops: _ListOrDict[SparseLabelOp | None] = _ListOrDict(second_q_ops)

        qubit_ops: _ListOrDict = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_ops[name] = PauliSumOp(self._map_clifford_single(second_q_op))

        returned_ops: PauliSumOp | ListOrDictType[PauliSumOp] = qubit_ops.unwrap(
            wrapped_type, suppress_none=suppress_none
        )

        return returned_ops

    def symmetry_reduce_clifford(
        self,
        second_q_ops: PauliSumOp | ListOrDictType[PauliSumOp],
        suppress_none: bool = False,
    ) -> PauliSumOp | ListOrDictType[PauliSumOp]:
        """Maps a second quantized operator or a list, dict of second quantized operators based on
        the current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof.
            suppress_none: If None should be placed in the output list where an operator
                did not commute with symmetry, to maintain order, or whether that should
                be suppressed where the output list length may then be smaller than the input.

        Returns:
            A qubit operator in the form of a PauliSumOp, or list (resp. dict) thereof if a list
            (resp. dict) of second quantized operators was supplied.
        """
        wrapped_type = type(second_q_ops)

        if issubclass(wrapped_type, PauliSumOp):
            second_q_ops = [second_q_ops]
            suppress_none = False

        wrapped_second_q_ops: _ListOrDict[PauliSumOp] = _ListOrDict(second_q_ops)

        qubit_ops: _ListOrDict[PauliSumOp] = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            qubit_op = self._symmetry_reduce_clifford_single(second_q_op.primitive)
            if self.check_commutes:
                if qubit_op is not None:
                    qubit_ops[name] = PauliSumOp(qubit_op)
            else:
                qubit_ops[name] = PauliSumOp(qubit_op)

        returned_ops: PauliSumOp | ListOrDictType[PauliSumOp] = qubit_ops.unwrap(
            wrapped_type, suppress_none=suppress_none
        )

        return returned_ops

    @staticmethod
    def _check_commutes(sq_paulis: PauliList, qubit_op: SparsePauliOp) -> bool:
        # commutes = []
        commuting_rows = qubit_op.paulis.commutes_with_all(sq_paulis)
        commutes = len(commuting_rows) == qubit_op.size
        logger.debug("  '%s' commutes: %s", id(qubit_op), commutes)
        return commutes
