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

"""TODO."""

from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from .vibrational_integrals import (
    _VibrationalIntegrals,
    BosonicBasis,
)
from .property import Property


class OccupiedModals(Property):
    """TODO."""

    def __init__(
        self,
        basis: Optional[BosonicBasis] = None,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._basis = basis
        self._integrals: _VibrationalIntegrals = None

    @property
    def basis(self) -> BosonicBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: BosonicBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "OccupiedModals":
        """TODO."""
        if isinstance(result, QMolecule):
            raise QiskitNatureError("TODO.")

        w_h = cast(WatsonHamiltonian, result)

        return cls()

    def second_q_ops(self) -> List[VibrationalOp]:
        """TODO."""
        ops = []
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        for mode in range(num_modes):
            ops.append(self.get_mode_op(mode))
        return ops

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass

    def get_mode_op(self, mode) -> VibrationalOp:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros((num_modes, max_num_modals, max_num_modals))

        for modal in range(num_modals_per_mode[mode]):
            matrix[(mode, modal, modal)] = 1.0

        labels = self._create_num_body_labels(matrix)
        return VibrationalOp(labels, num_modes, num_modals_per_mode)

    @staticmethod
    def _create_num_body_labels(matrix: np.ndarray) -> List[Tuple[str, complex]]:
        num_body_labels = []
        nonzero = np.nonzero(matrix)
        for coeff, indices in zip(matrix[nonzero], zip(*nonzero)):
            grouped_indices = sorted(
                [tuple(int(j) for j in indices[i : i + 3]) for i in range(0, len(indices), 3)]
            )
            coeff_label = _VibrationalIntegrals._create_label_for_coeff(grouped_indices)
            num_body_labels.append((coeff_label, coeff))
        return num_body_labels

    @staticmethod
    def _create_label_for_coeff(indices: List[Tuple[int, ...]]) -> str:
        complete_labels_list = []
        for mode, modal_raise, modal_lower in indices:
            if modal_raise <= modal_lower:
                complete_labels_list.append(f"+_{mode}*{modal_raise}")
                complete_labels_list.append(f"-_{mode}*{modal_lower}")
            else:
                complete_labels_list.append(f"-_{mode}*{modal_lower}")
                complete_labels_list.append(f"+_{mode}*{modal_raise}")
        complete_label = " ".join(complete_labels_list)
        return complete_label
