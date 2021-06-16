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

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers.second_quantization import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from .bases import VibrationalBasis
from ..property import Property


class OccupiedModals(Property):
    """The OccupiedModals property."""

    def __init__(
        self,
        basis: Optional[VibrationalBasis] = None,
    ):
        """
        Args:
            basis: the ``VibrationalBasis`` through which to map the integrals into second
                quantization. This property **MUST** be set before the second-quantized operator can
                be constructed.
        """
        super().__init__(self.__class__.__name__)
        self._basis = basis

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "OccupiedModals":
        """Construct an OccupiedModals instance from a WatsonHamiltonian.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                WatsonHamiltonian is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a QMolecule is provided.
        """
        cls._validate_input_type(result, WatsonHamiltonian)

        return cls()

    def second_q_ops(self) -> List[VibrationalOp]:
        """Returns a list of operators each evaluating the occupied modal on a mode."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        ops = []
        for mode in range(num_modes):
            ops.append(self._get_mode_op(mode))

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

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an `qiskit_nature.result.EigenstateResult` in the context of this Property.

        This is currently a method stub which may be used in the future.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
