# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The vibrational properties container."""

from __future__ import annotations

from typing import cast

from qiskit_nature.second_q.properties import OccupiedModals

from .properties_container import PropertiesContainer


class VibrationalPropertiesContainer(PropertiesContainer):
    """The container class for vibrational structure properties."""

    @property
    def occupied_modals(self) -> OccupiedModals | None:
        """Returns the occupied modals property."""
        return cast(OccupiedModals, self._getter(OccupiedModals))

    @occupied_modals.setter
    def occupied_modals(self, occupied_modals: OccupiedModals | None) -> None:
        """Sets the occupied modals property."""
        self._setter(occupied_modals, OccupiedModals)
