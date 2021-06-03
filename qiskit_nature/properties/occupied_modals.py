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

from typing import Dict, List, Optional, Tuple, Union

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

        return cls()

    def second_q_ops(self) -> List[VibrationalOp]:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        ops = []
        for mode in range(num_modes):
            ops.append(self.get_mode_op(mode))

        return ops

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass

    def get_mode_op(self, mode) -> VibrationalOp:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode

        labels: List[Tuple[str, complex]] = []

        for modal in range(num_modals_per_mode[mode]):
            labels.append((f"+_{mode}*{modal} -_{mode}*{modal}", 1.0))

        return VibrationalOp(labels, len(num_modals_per_mode), num_modals_per_mode)
