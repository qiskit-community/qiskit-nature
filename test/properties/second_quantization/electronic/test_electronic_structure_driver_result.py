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

"""Test ElectronicStructureDriverResult Property"""

from test import QiskitNatureTestCase

import h5py
import numpy as np

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.properties.second_quantization.driver_metadata import DriverMetadata
from qiskit_nature.properties.second_quantization.electronic import (
    AngularMomentum,
    DipoleMoment,
    ElectronicDipoleMoment,
    ElectronicEnergy,
    ElectronicStructureDriverResult,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasisTransform
from qiskit_nature.properties.second_quantization.electronic.integrals import ElectronicIntegrals


class TestElectronicStructureDriverResult(QiskitNatureTestCase):
    """Test ElectronicStructureDriverResult Property"""

    def compare_angular_momentum(
        self, first: AngularMomentum, second: AngularMomentum, msg: str = None
    ) -> None:
        """Compares two AngularMomentum instances."""
        if first._num_spin_orbitals != second._num_spin_orbitals:
            raise self.failureException(msg)
        if first._spin != second._spin:
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
        if first._axis != second._axis:
            raise self.failureException(msg)

        for f_basis, s_basis in zip(
            first._electronic_integrals.values(), second._electronic_integrals.values()
        ):
            for f_ints, s_ints in zip(f_basis.values(), s_basis.values()):
                self.compare_electronic_integral(f_ints, s_ints)

    def compare_electronic_energy(
        self, first: ElectronicEnergy, second: ElectronicEnergy, msg: str = None
    ) -> None:
        """Compares two ElectronicEnergy instances."""
        for f_basis, s_basis in zip(
            first._electronic_integrals.values(), second._electronic_integrals.values()
        ):
            for f_ints, s_ints in zip(f_basis.values(), s_basis.values()):
                self.compare_electronic_integral(f_ints, s_ints)

        if not np.isclose(first.nuclear_repulsion_energy, second.nuclear_repulsion_energy):
            raise self.failureException(msg)
        if not np.isclose(first.reference_energy, second.reference_energy):
            raise self.failureException(msg)
        if not np.allclose(first.orbital_energies, second.orbital_energies):
            raise self.failureException(msg)

        self.assertEqual(first.kinetic, second.kinetic)
        self.assertEqual(first.overlap, second.overlap)

    def compare_magnetization(
        self, first: Magnetization, second: Magnetization, msg: str = None
    ) -> None:
        """Compares two Magnetization instances."""
        if first._num_spin_orbitals != second._num_spin_orbitals:
            raise self.failureException(msg)

    def compare_particle_number(
        self, first: ParticleNumber, second: ParticleNumber, msg: str = None
    ) -> None:
        """Compares two ParticleNumber instances."""
        if first._num_spin_orbitals != second._num_spin_orbitals:
            raise self.failureException(msg)
        if first._num_alpha != second._num_alpha:
            raise self.failureException(msg)
        if first._num_beta != second._num_beta:
            raise self.failureException(msg)
        if not np.allclose(first._occupation_alpha, second._occupation_alpha):
            raise self.failureException(msg)
        if not np.allclose(first._occupation_beta, second._occupation_beta):
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
        if first._basis != second._basis:
            raise self.failureException(msg)
        if first._num_body_terms != second._num_body_terms:
            raise self.failureException(msg)
        if first._matrix_representations != second._matrix_representations:
            raise self.failureException(msg)
        for f_mat, s_mat in zip(first._matrices, second._matrices):
            if not np.allclose(f_mat, s_mat):
                raise self.failureException(msg)

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

        driver = HDF5Driver(
            self.get_resource_path(
                "BeH_sto3g_reduced.hdf5", "transformers/second_quantization/electronic"
            )
        )
        self.expected = driver.run()

    def test_from_hdf5(self):
        """Test from_hdf5."""
        with h5py.File(
            self.get_resource_path(
                "electronic_structure_driver_result.hdf5",
                "properties/second_quantization/electronic/resources",
            ),
            "r",
        ) as file:
            for group in file.values():
                prop = ElectronicStructureDriverResult.from_hdf5(group)
                for inner_prop in iter(prop):
                    expected = self.expected.get_property(type(inner_prop))
                    self.assertEqual(inner_prop, expected)
