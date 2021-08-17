# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The OccupiedModals property."""

from typing import List, Optional, Tuple, Union

from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .bases import VibrationalBasis
from .types import VibrationalProperty


class OccupiedModals(VibrationalProperty):
    """The OccupiedModals property."""

    def __init__(
        self,
        basis: Optional[VibrationalBasis] = None,
    ) -> None:
        """
        Args:
            basis: the
                :class:`~qiskit_nature.properties.second_quantization.vibrational.bases.VibrationalBasis`
                through which to map the integrals into second quantization. This attribute **MUST**
                be set before the second-quantized operator can be constructed.
        """
        super().__init__(self.__class__.__name__, basis)

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "OccupiedModals":
        """Construct an OccupiedModals instance from a
        :class:`~qiskit_nature.drivers.WatsonHamiltonian`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.drivers.WatsonHamiltonian` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.QMolecule` is provided.
        """
        cls._validate_input_type(result, WatsonHamiltonian)

        return cls()

    def second_q_ops(self) -> List[VibrationalOp]:
        """Returns a list of operators each evaluating the occupied modal on a mode."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        ops = [self._get_mode_op(mode) for mode in range(num_modes)]
        return ops

    def _get_mode_op(self, mode: int) -> VibrationalOp:
        """Constructs an operator to evaluate which modal of a given mode is occupied.

        Args:
            mode: the mode index.

        Returns:
            The operator to evaluate which modal of the given mode is occupied.
        """
        num_modals_per_mode = self.basis._num_modals_per_mode

        labels: List[Tuple[str, complex]] = []

        for modal in range(num_modals_per_mode[mode]):
            labels.append((f"+_{mode}*{modal} -_{mode}*{modal}", 1.0))

        return VibrationalOp(labels, len(num_modals_per_mode), num_modals_per_mode)

    # TODO: refactor after closing https://github.com/Qiskit/qiskit-terra/issues/6772
    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.num_occupied_modals_per_mode = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore

        num_modes = len(self._basis._num_modals_per_mode)

        for aux_op_eigenvalues in aux_operator_eigenvalues:
            occ_modals = []
            for mode in range(num_modes):
                if aux_op_eigenvalues[mode] is not None:
                    occ_modals.append(aux_op_eigenvalues[mode][0].real)  # type: ignore
                else:
                    occ_modals.append(None)
            result.num_occupied_modals_per_mode.append(occ_modals)  # type: ignore
