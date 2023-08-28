# This code is part of a Qiskit project.
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

"""The OccupiedModals property."""

from __future__ import annotations

from typing import Mapping, Sequence

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import VibrationalOp


class OccupiedModals:
    """The OccupiedModals property.

    The following attributes can be set via the initializer but can also be read and updated once
    the ``OccupiedModals`` object has been constructed.

    Attributes:
        num_modals (Sequence[int]): the number of modals per mode in the system.
    """

    def __init__(self, num_modals: Sequence[int]) -> None:
        """
        Args:
            num_modals: the number of modals for each mode.
        """
        self.num_modals = num_modals

    def second_q_ops(self) -> Mapping[str, VibrationalOp]:
        """Returns the second quantized operators indicating the occupied modals per mode.

        Returns:
            A mapping of strings to `VibrationalOp` objects.
        """
        return {str(mode): self._get_mode_op(mode) for mode in range(len(self.num_modals))}

    def _get_mode_op(self, mode: int) -> VibrationalOp:
        """Constructs an operator to evaluate which modal of a given mode is occupied.

        Args:
            mode: the mode index.

        Returns:
            The operator to evaluate which modal of the given mode is occupied.
        """
        labels: dict[str, complex] = {}

        for modal in range(self.num_modals[mode]):
            labels[f"+_{mode}_{modal} -_{mode}_{modal}"] = 1.0

        return VibrationalOp(labels, self.num_modals)

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.num_occupied_modals_per_mode = []

        if not isinstance(result.aux_operators_evaluated, list):
            aux_operators_evaluated = [result.aux_operators_evaluated]
        else:
            aux_operators_evaluated = result.aux_operators_evaluated

        for aux_op_eigenvalues in aux_operators_evaluated:
            occ_modals = []
            for mode, _ in enumerate(self.num_modals):
                _key = str(mode) if isinstance(aux_op_eigenvalues, dict) else mode
                if aux_op_eigenvalues[_key] is not None:
                    occ_modals.append(aux_op_eigenvalues[_key].real)
                else:
                    occ_modals.append(None)
            result.num_occupied_modals_per_mode.append(occ_modals)
