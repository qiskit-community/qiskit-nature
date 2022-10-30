# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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

from typing import Optional, Mapping

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import VibrationalOp

from .bases import VibrationalBasis


class OccupiedModals:
    """The OccupiedModals property."""

    def __init__(
        self,
        basis: Optional[VibrationalBasis] = None,
    ) -> None:
        """
        Args:
            basis: the
                :class:`~qiskit_nature.second_q.properties.bases.VibrationalBasis`
                through which to map the integrals into second quantization. This attribute **MUST**
                be set before the second-quantized operator can be constructed.
        """
        self._basis: VibrationalBasis = basis

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    def second_q_ops(self) -> Mapping[str, VibrationalOp]:
        """Returns the second quantized operators indicating the occupied modals per mode.

        Returns:
            A mapping of strings to `VibrationalOp` objects.
        """
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        return {str(mode): self._get_mode_op(mode) for mode in range(num_modes)}

    def _get_mode_op(self, mode: int) -> VibrationalOp:
        """Constructs an operator to evaluate which modal of a given mode is occupied.

        Args:
            mode: the mode index.

        Returns:
            The operator to evaluate which modal of the given mode is occupied.
        """
        num_modals_per_mode = self.basis._num_modals_per_mode

        labels: dict[str, complex] = {}

        for modal in range(num_modals_per_mode[mode]):
            labels[f"+_{mode}_{modal} -_{mode}_{modal}"] = 1.0

        return VibrationalOp(labels, num_modals_per_mode)

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

        num_modes = len(self._basis._num_modals_per_mode)

        for aux_op_eigenvalues in aux_operators_evaluated:
            occ_modals = []
            for mode in range(num_modes):
                _key = str(mode) if isinstance(aux_op_eigenvalues, dict) else mode
                if aux_op_eigenvalues[_key] is not None:
                    occ_modals.append(aux_op_eigenvalues[_key].real)
                else:
                    occ_modals.append(None)
            result.num_occupied_modals_per_mode.append(occ_modals)
