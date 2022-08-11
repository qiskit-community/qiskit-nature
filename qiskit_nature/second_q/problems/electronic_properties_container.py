# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The electronic properties container."""

from __future__ import annotations

from typing import cast

from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDipoleMoment,
    Magnetization,
    ParticleNumber,
)

from .properties_container import PropertiesContainer


class ElectronicPropertiesContainer(PropertiesContainer):
    """The container class for electronic structure properties."""

    @property
    def angular_momentum(self) -> AngularMomentum | None:
        """Returns the :class:`qiskit_nature.second_q.properties.AngularMomentum` property."""
        return cast(AngularMomentum, self._properties.get("AngularMomentum", None))

    @angular_momentum.setter
    def angular_momentum(self, angular_momentum: AngularMomentum | None) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.AngularMomentum` property."""
        self._setter(angular_momentum, AngularMomentum)

    @property
    def electronic_dipole_moment(self) -> ElectronicDipoleMoment | None:
        """Returns the :class:`qiskit_nature.second_q.properties.ElectronicDipoleMoment` property."""
        return cast(ElectronicDipoleMoment, self._properties.get("ElectronicDipoleMoment", None))

    @electronic_dipole_moment.setter
    def electronic_dipole_moment(
        self, electronic_dipole_moment: ElectronicDipoleMoment | None
    ) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.ElectronicDipoleMoment` property."""
        self._setter(electronic_dipole_moment, ElectronicDipoleMoment)

    @property
    def magnetization(self) -> Magnetization | None:
        """Returns the :class:`qiskit_nature.second_q.properties.Magnetization` property."""
        return cast(Magnetization, self._properties.get("Magnetization", None))

    @magnetization.setter
    def magnetization(self, magnetization: Magnetization | None) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.Magnetization` property."""
        self._setter(magnetization, Magnetization)

    @property
    def particle_number(self) -> ParticleNumber | None:
        """Returns the :class:`qiskit_nature.second_q.properties.ParticleNumber` property."""
        return cast(ParticleNumber, self._properties.get("ParticleNumber", None))

    @particle_number.setter
    def particle_number(self, particle_number: ParticleNumber | None) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.ParticleNumber` property."""
        self._setter(particle_number, ParticleNumber)
