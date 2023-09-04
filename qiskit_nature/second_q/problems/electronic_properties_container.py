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

"""The electronic properties container."""

from __future__ import annotations

from typing import cast

from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    ElectronicDipoleMoment,
    Magnetization,
    ParticleNumber,
)

from .properties_container import PropertiesContainer


class ElectronicPropertiesContainer(PropertiesContainer):
    """The container class for electronic structure properties."""

    @property
    def angular_momentum(self) -> AngularMomentum | None:
        """Returns the angular momentum property."""
        return cast(AngularMomentum, self._getter(AngularMomentum))

    @angular_momentum.setter
    def angular_momentum(self, angular_momentum: AngularMomentum | None) -> None:
        """Sets the angular momentum property."""
        self._setter(angular_momentum, AngularMomentum)

    @property
    def electronic_density(self) -> ElectronicDensity | None:
        """Returns the electronic density property."""
        return cast(ElectronicDensity, self._getter(ElectronicDensity))

    @electronic_density.setter
    def electronic_density(self, electronic_density: ElectronicDensity | None) -> None:
        """Sets the electronic density property."""
        self._setter(electronic_density, ElectronicDensity)

    @property
    def electronic_dipole_moment(self) -> ElectronicDipoleMoment | None:
        """Returns the electronic dipole moment property."""
        return cast(ElectronicDipoleMoment, self._getter(ElectronicDipoleMoment))

    @electronic_dipole_moment.setter
    def electronic_dipole_moment(
        self, electronic_dipole_moment: ElectronicDipoleMoment | None
    ) -> None:
        """Sets the electronic dipole moment property."""
        self._setter(electronic_dipole_moment, ElectronicDipoleMoment)

    @property
    def magnetization(self) -> Magnetization | None:
        """Returns the magnetization property."""
        return cast(Magnetization, self._getter(Magnetization))

    @magnetization.setter
    def magnetization(self, magnetization: Magnetization | None) -> None:
        """Sets the magnetization property."""
        self._setter(magnetization, Magnetization)

    @property
    def particle_number(self) -> ParticleNumber | None:
        """Returns the particle number property."""
        return cast(ParticleNumber, self._getter(ParticleNumber))

    @particle_number.setter
    def particle_number(self, particle_number: ParticleNumber | None) -> None:
        """Sets the particle number property."""
        self._setter(particle_number, ParticleNumber)
