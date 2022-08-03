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
        if angular_momentum is None:
            self._properties.pop("AngularMomentum", None)
            return

        if not isinstance(angular_momentum, AngularMomentum):
            raise TypeError(
                f"Only objects of type 'AngularMomentum' are supported, not {type(angular_momentum)}."
            )
        self._properties["AngularMomentum"] = angular_momentum

    @property
    def electronic_dipole_moment(self) -> ElectronicDipoleMoment | None:
        """Returns the :class:`qiskit_nature.second_q.properties.ElectronicDipoleMoment` property."""
        return cast(ElectronicDipoleMoment, self._properties.get("ElectronicDipoleMoment", None))

    @electronic_dipole_moment.setter
    def electronic_dipole_moment(
        self, electronic_dipole_moment: ElectronicDipoleMoment | None
    ) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.ElectronicDipoleMoment` property."""
        if electronic_dipole_moment is None:
            self._properties.pop("ElectronicDipoleMoment", None)
            return

        if not isinstance(electronic_dipole_moment, ElectronicDipoleMoment):
            raise TypeError(
                "Only objects of type 'ElectronicDipoleMoment' are supported, not "
                f"{type(electronic_dipole_moment)}."
            )
        self._properties["ElectronicDipoleMoment"] = electronic_dipole_moment

    @property
    def magnetization(self) -> Magnetization | None:
        """Returns the :class:`qiskit_nature.second_q.properties.Magnetization` property."""
        return cast(Magnetization, self._properties.get("Magnetization", None))

    @magnetization.setter
    def magnetization(self, magnetization: Magnetization | None) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.Magnetization` property."""
        if magnetization is None:
            self._properties.pop("Magnetization", None)
            return

        if not isinstance(magnetization, Magnetization):
            raise TypeError(
                f"Only objects of type 'Magnetization' are supported, not {type(magnetization)}."
            )
        self._properties["Magnetization"] = magnetization

    @property
    def particle_number(self) -> ParticleNumber | None:
        """Returns the :class:`qiskit_nature.second_q.properties.ParticleNumber` property."""
        return cast(ParticleNumber, self._properties.get("ParticleNumber", None))

    @particle_number.setter
    def particle_number(self, particle_number: ParticleNumber | None) -> None:
        """Sets the :class:`qiskit_nature.second_q.properties.ParticleNumber` property."""
        if particle_number is None:
            self._properties.pop("ParticleNumber", None)
            return

        if not isinstance(particle_number, ParticleNumber):
            raise TypeError(
                f"Only objects of type 'ParticleNumber' are supported, not {type(particle_number)}."
            )
        self._properties["ParticleNumber"] = particle_number
