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

"""The VibrationalStructureDriverResult class."""

from typing import Dict, List, cast

from .... import ListOrDictType
from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp

from ..second_quantized_property import LegacyDriverResult
from .occupied_modals import OccupiedModals
from .vibrational_energy import VibrationalEnergy
from .types import GroupedVibrationalProperty


class VibrationalStructureDriverResult(GroupedVibrationalProperty):
    """The VibrationalStructureDriverResult class.

    This is a :class:`~qiskit_nature.properties.GroupedProperty` gathering all property objects
    previously stored in Qiskit Nature's :class:`~qiskit_nature.drivers.WatsonHamiltonian` object.
    """

    def __init__(self) -> None:
        """
        Property objects should be added via ``add_property`` rather than via the initializer.
        """
        super().__init__(self.__class__.__name__)
        self._num_modes: int = None

    @property
    def num_modes(self) -> int:
        """Returns the number of modes."""
        return self._num_modes

    @num_modes.setter
    def num_modes(self, num_modes: int) -> None:
        """Sets the number of modes."""
        self._num_modes = num_modes

    @classmethod
    def from_legacy_driver_result(
        cls, result: LegacyDriverResult
    ) -> "VibrationalStructureDriverResult":
        """Converts a :class:`~qiskit_nature.drivers.WatsonHamiltonian` into an
        ``VibrationalStructureDriverResult``.

        Args:
            result: the :class:`~qiskit_nature.drivers.WatsonHamiltonian` to convert.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.QMolecule` is provided.
        """
        cls._validate_input_type(result, WatsonHamiltonian)

        ret = cls()

        watson = cast(WatsonHamiltonian, result)

        ret.num_modes = watson.num_modes
        ret.add_property(VibrationalEnergy.from_legacy_driver_result(watson))
        ret.add_property(OccupiedModals.from_legacy_driver_result(watson))

        return ret

    def second_q_ops(self, return_list: bool = True) -> ListOrDictType[VibrationalOp]:
        """Returns a list or dictionary of
        :class:`~qiskit_nature.operators.second_quantization.VibrationalOp`s given by the properties
        contained in this one."""
        ops: ListOrDictType[VibrationalOp]
        if return_list:
            ops = []
            for cls in [VibrationalEnergy, OccupiedModals]:
                prop = self.get_property(cls)  # type: ignore
                if prop is None:
                    continue
                ops.extend(prop.second_q_ops(return_list))
            return ops

        ops = {}
        for prop in iter(self):
            ops.update(prop.second_q_ops(return_list))
        return ops
