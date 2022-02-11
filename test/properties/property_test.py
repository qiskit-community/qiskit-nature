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

"""PropertyTest class"""

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.properties.second_quantization.driver_metadata import DriverMetadata
from qiskit_nature.properties.second_quantization.electronic import (
    AngularMomentum,
    DipoleMoment,
    ElectronicDipoleMoment,
    ElectronicEnergy,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasisTransform
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.properties.second_quantization.vibrational import (
    OccupiedModals,
    VibrationalEnergy,
)
from qiskit_nature.properties.second_quantization.vibrational.integrals import VibrationalIntegrals


class PropertyTest(QiskitNatureTestCase):
    """Property instance tester"""

    def compare_angular_momentum(
        self, first: AngularMomentum, second: AngularMomentum, msg: str = None
    ) -> None:
        """Compares two AngularMomentum instances."""
        if first.num_spin_orbitals != second.num_spin_orbitals:
            raise self.failureException(msg)
        if first.spin != second.spin:
            raise self.failureException(msg)
        if not np.isclose(first.absolute_tolerance, second.absolute_tolerance):
            raise self.failureException(msg)
        if not np.isclose(first.relative_tolerance, second.relative_tolerance):
            raise self.failureException(msg)

    def compare_electronic_dipole_moment(
        self, first: ElectronicDipoleMoment, second: ElectronicDipoleMoment, msg: str = None
    ) -> None:
        """Compares two ElectronicDipoleMoment instances."""
        for f_axis in iter(first):
            s_axis = second.get_property(f_axis.name)
            self.assertEqual(f_axis, s_axis)

        if first.reverse_dipole_sign != second.reverse_dipole_sign:
            raise self.failureException(msg)

        if not np.allclose(first.nuclear_dipole_moment, second.nuclear_dipole_moment):
            raise self.failureException(msg)

    def compare_dipole_moment(
        self, first: DipoleMoment, second: DipoleMoment, msg: str = None
    ) -> None:
        """Compares two DipoleMoment instances."""
        if first.axis != second.axis:
            raise self.failureException(msg)

        for f_ints, s_ints in zip(first, second):
            self.compare_electronic_integral(f_ints, s_ints)

    def compare_electronic_energy(
        self, first: ElectronicEnergy, second: ElectronicEnergy, msg: str = None
    ) -> None:
        """Compares two ElectronicEnergy instances."""
        for f_ints, s_ints in zip(first, second):
            self.compare_electronic_integral(f_ints, s_ints)

        if not np.isclose(first.nuclear_repulsion_energy, second.nuclear_repulsion_energy):
            raise self.failureException(msg)
        if not np.isclose(first.reference_energy, second.reference_energy):
            raise self.failureException(msg)
        if not np.allclose(first.orbital_energies, second.orbital_energies):
            raise self.failureException(msg)

        self.assertEqual(first.overlap, second.overlap)
        self.assertEqual(first.kinetic, second.kinetic)

    def compare_magnetization(
        self, first: Magnetization, second: Magnetization, msg: str = None
    ) -> None:
        """Compares two Magnetization instances."""
        if first.num_spin_orbitals != second.num_spin_orbitals:
            raise self.failureException(msg)

    def compare_particle_number(
        self, first: ParticleNumber, second: ParticleNumber, msg: str = None
    ) -> None:
        """Compares two ParticleNumber instances."""
        if first.num_spin_orbitals != second.num_spin_orbitals:
            raise self.failureException(msg)
        if first.num_alpha != second.num_alpha:
            raise self.failureException(msg)
        if first.num_beta != second.num_beta:
            raise self.failureException(msg)
        if not np.allclose(first.occupation_alpha, second.occupation_alpha):
            raise self.failureException(msg)
        if not np.allclose(first.occupation_beta, second.occupation_beta):
            raise self.failureException(msg)
        if not np.isclose(first.absolute_tolerance, second.absolute_tolerance):
            raise self.failureException(msg)
        if not np.isclose(first.relative_tolerance, second.relative_tolerance):
            raise self.failureException(msg)

    def compare_driver_metadata(
        self, first: DriverMetadata, second: DriverMetadata, msg: str = None
    ) -> None:
        """Compares two DriverMetadata instances."""
        if first.program != second.program:
            raise self.failureException(msg)
        if first.version != second.version:
            raise self.failureException(msg)
        if first.config != second.config:
            raise self.failureException(msg)

    def compare_electronic_basis_transform(
        self, first: ElectronicBasisTransform, second: ElectronicBasisTransform, msg: str = None
    ) -> None:
        """Compares two ElectronicBasisTransform instances."""
        if first.initial_basis != second.initial_basis:
            raise self.failureException(msg)
        if first.final_basis != second.final_basis:
            raise self.failureException(msg)
        if not np.allclose(first.coeff_alpha, second.coeff_alpha):
            raise self.failureException(msg)
        if not np.allclose(first.coeff_beta, second.coeff_beta):
            raise self.failureException(msg)

    def compare_electronic_integral(
        self, first: ElectronicIntegrals, second: ElectronicIntegrals, msg: str = None
    ) -> None:
        """Compares two ElectronicIntegrals instances."""
        if first.name != second.name:
            raise self.failureException(msg)
        if first.basis != second.basis:
            raise self.failureException(msg)
        if first.num_body_terms != second.num_body_terms:
            raise self.failureException(msg)
        if not np.isclose(first.threshold, second.threshold):
            raise self.failureException(msg)
        for f_mat, s_mat in zip(first, second):
            if f_mat is None:
                self.assertIsNone(s_mat)
                continue
            if not np.allclose(f_mat, s_mat):
                raise self.failureException(msg)

    def compare_vibrational_integral(
        self, first: VibrationalIntegrals, second: VibrationalIntegrals, msg: str = None
    ) -> None:
        """Compares two VibrationalIntegral instances."""
        if first.name != second.name:
            raise self.failureException(msg)

        if first.num_body_terms != second.num_body_terms:
            raise self.failureException(msg)

        for f_int, s_int in zip(first.integrals, second.integrals):
            if not np.isclose(f_int[0], s_int[0]):
                raise self.failureException(msg)

            if not all(f == s for f, s in zip(f_int[1:], s_int[1:])):
                raise self.failureException(msg)

    def compare_vibrational_energy(
        self, first: VibrationalEnergy, second: VibrationalEnergy, msg: str = None
    ) -> None:
        # pylint: disable=unused-argument
        """Compares two VibrationalEnergy instances."""
        for f_ints, s_ints in zip(first, second):
            self.compare_vibrational_integral(f_ints, s_ints)

    def compare_occupied_modals(
        self, first: OccupiedModals, second: OccupiedModals, msg: str = None
    ) -> None:
        # pylint: disable=unused-argument
        """Compares two OccupiedModals instances."""
        pass

    def setUp(self) -> None:
        """Setup expected object."""
        super().setUp()
        self.addTypeEqualityFunc(AngularMomentum, self.compare_angular_momentum)
        self.addTypeEqualityFunc(DipoleMoment, self.compare_dipole_moment)
        self.addTypeEqualityFunc(ElectronicDipoleMoment, self.compare_electronic_dipole_moment)
        self.addTypeEqualityFunc(ElectronicEnergy, self.compare_electronic_energy)
        self.addTypeEqualityFunc(Magnetization, self.compare_magnetization)
        self.addTypeEqualityFunc(ParticleNumber, self.compare_particle_number)
        self.addTypeEqualityFunc(DriverMetadata, self.compare_driver_metadata)
        self.addTypeEqualityFunc(ElectronicBasisTransform, self.compare_electronic_basis_transform)
        self.addTypeEqualityFunc(ElectronicIntegrals, self.compare_electronic_integral)
        self.addTypeEqualityFunc(OneBodyElectronicIntegrals, self.compare_electronic_integral)
        self.addTypeEqualityFunc(TwoBodyElectronicIntegrals, self.compare_electronic_integral)
        self.addTypeEqualityFunc(VibrationalIntegrals, self.compare_vibrational_integral)
        self.addTypeEqualityFunc(VibrationalEnergy, self.compare_vibrational_energy)
        self.addTypeEqualityFunc(OccupiedModals, self.compare_occupied_modals)
