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

"""The VibrationalDriverResult class."""

from typing import List, cast

from qiskit_nature.drivers.second_quantization import WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp

from ..driver_result import DriverResult
from ..second_quantized_property import (LegacyDriverResult,
                                         LegacyVibrationalDriverResult)
from .bases import VibrationalBasis
from .occupied_modals import OccupiedModals
from .vibrational_energy import VibrationalEnergy
from .vibrational_property import VibrationalProperty


# Pylint<2.9.0 raises false-positive no-member errors (E1101) for abstract properties
# pylint: disable=no-member
class VibrationalDriverResult(DriverResult[VibrationalProperty]):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._num_modes: int = None

    @property
    def num_modes(self) -> int:
        """Returns the num_modes."""
        return self._num_modes

    @num_modes.setter
    def num_modes(self, num_modes: int) -> None:
        """Sets the num_modes."""
        self._num_modes = num_modes

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return list(self._properties.values())[0].basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        for prop in self._properties.values():
            prop.basis = basis

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "VibrationalDriverResult":
        """TODO."""
        cls._validate_input_type(result, LegacyVibrationalDriverResult)

        ret = cls()

        watson = cast(WatsonHamiltonian, result)

        ret.num_modes = watson.num_modes
        ret.add_property(VibrationalEnergy.from_legacy_driver_result(watson))
        ret.add_property(OccupiedModals.from_legacy_driver_result(watson))

        return ret

    def second_q_ops(self) -> List[VibrationalOp]:
        """TODO."""
        ops: List[VibrationalOp] = []
        # TODO: make aux_ops a Dict? Then we don't need to hard-code the order of these properties.
        for cls in [VibrationalEnergy, OccupiedModals]:
            prop = self.get_property(cls)
            if prop is None:
                continue
            ops.extend(prop.second_q_ops())
        return ops
