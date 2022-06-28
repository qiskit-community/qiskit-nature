# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver """

from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from qiskit_nature.drivers import Molecule
from qiskit_nature.second_quantization.operator_factories.electronic import (
    ParticleNumber,
    ElectronicEnergy,
    ElectronicDipoleMoment,
    ElectronicStructureDriverResult,
)
from qiskit_nature.second_quantization.operator_factories.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)


class TestDriver(ABC):
    """Common driver tests. For H2 @ 0.735, sto3g"""

    MOLECULE = Molecule(
        geometry=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.735])],
        multiplicity=1,
        charge=0,
    )

    def __init__(self):
        self.log = None
        self.driver_result: ElectronicStructureDriverResult = None

    @abstractmethod
    def subTest(self, msg, **kwargs):
        # pylint: disable=invalid-name
        """subtest"""
        raise Exception("Abstract method")

    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """assert Almost Equal"""
        raise Exception("Abstract method")

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """assert equal"""
        raise Exception("Abstract method")

    @abstractmethod
    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        """assert Sequence Equal"""
        raise Exception("Abstract method")

    def test_driver_result_electronic_energy(self):
        """Test the ElectronicEnergy property."""
        electronic_energy = cast(
            ElectronicEnergy, self.driver_result.get_property(ElectronicEnergy)
        )

        with self.subTest("reference energy"):
            self.log.debug("HF energy: %s", electronic_energy.reference_energy)
            self.assertAlmostEqual(electronic_energy.reference_energy, -1.117, places=3)

        with self.subTest("nuclear repulsion energy"):
            self.log.debug(
                "Nuclear repulsion energy: %s", electronic_energy.nuclear_repulsion_energy
            )
            self.assertAlmostEqual(electronic_energy.nuclear_repulsion_energy, 0.72, places=2)

        with self.subTest("orbital energies"):
            self.log.debug("orbital energies %s", electronic_energy.orbital_energies)
            np.testing.assert_array_almost_equal(
                electronic_energy.orbital_energies, [-0.5806, 0.6763], decimal=4
            )

        with self.subTest("1-body integrals"):
            mo_onee_ints = electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1)
            self.log.debug("MO one electron integrals %s", mo_onee_ints)
            self.assertEqual(mo_onee_ints._matrices[0].shape, (2, 2))
            np.testing.assert_array_almost_equal(
                np.absolute(mo_onee_ints._matrices[0]),
                [[1.2563, 0.0], [0.0, 0.4719]],
                decimal=4,
            )

        with self.subTest("2-body integrals"):
            mo_eri_ints = electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
            self.log.debug("MO two electron integrals %s", mo_eri_ints)
            self.assertEqual(mo_eri_ints._matrices[0].shape, (2, 2, 2, 2))
            np.testing.assert_array_almost_equal(
                np.absolute(mo_eri_ints._matrices[0]),
                [
                    [[[0.6757, 0.0], [0.0, 0.6646]], [[0.0, 0.1809], [0.1809, 0.0]]],
                    [[[0.0, 0.1809], [0.1809, 0.0]], [[0.6646, 0.0], [0.0, 0.6986]]],
                ],
                decimal=4,
            )

    def test_driver_result_particle_number(self):
        """Test the ParticleNumber property."""
        particle_number = cast(ParticleNumber, self.driver_result.get_property(ParticleNumber))

        with self.subTest("orbital number"):
            self.log.debug("Number of orbitals is %s", particle_number.num_spin_orbitals)
            self.assertEqual(particle_number.num_spin_orbitals, 4)

        with self.subTest("alpha electron number"):
            self.log.debug("Number of alpha electrons is %s", particle_number.num_alpha)
            self.assertEqual(particle_number.num_alpha, 1)

        with self.subTest("beta electron number"):
            self.log.debug("Number of beta electrons is %s", particle_number.num_beta)
            self.assertEqual(particle_number.num_beta, 1)

    def test_driver_result_molecule(self):
        """Test the Molecule object."""
        molecule = self.driver_result.molecule

        with self.subTest("molecular charge"):
            self.log.debug("molecular charge is %s", molecule.charge)
            self.assertEqual(molecule.charge, 0)

        with self.subTest("multiplicity"):
            self.log.debug("multiplicity is %s", molecule.multiplicity)
            self.assertEqual(molecule.multiplicity, 1)

        with self.subTest("atom number"):
            self.log.debug("num atoms %s", len(molecule.geometry))
            self.assertEqual(len(molecule.geometry), 2)

        with self.subTest("atoms"):
            self.log.debug("atom symbol %s", molecule.atoms)
            self.assertSequenceEqual(molecule.atoms, ["H", "H"])

        with self.subTest("coordinates"):
            coords = [coord for _, coord in molecule.geometry]
            self.log.debug("atom xyz %s", coords)
            np.testing.assert_array_almost_equal(
                coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]], decimal=4
            )

    def test_driver_result_basis_transform(self):
        """Test the ElectronicBasisTransform object."""
        basis_transform = cast(
            ElectronicBasisTransform, self.driver_result.get_property(ElectronicBasisTransform)
        )

        self.log.debug("MO coeffs xyz %s", basis_transform.coeff_alpha)
        self.assertEqual(basis_transform.coeff_alpha.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            np.absolute(basis_transform.coeff_alpha),
            [[0.5483, 1.2183], [0.5483, 1.2183]],
            decimal=4,
        )

    def test_driver_result_electronic_dipole(self):
        """Test the ElectronicDipoleMoment property."""
        dipole = self.driver_result.get_property(ElectronicDipoleMoment)

        self.log.debug("has dipole integrals %s", dipole is not None)
        if dipole is not None:
            dipole = cast(ElectronicDipoleMoment, dipole)

            with self.subTest("x axis"):
                mo_x_dip_ints = dipole.get_property("DipoleMomentX").get_electronic_integral(
                    ElectronicBasis.MO, 1
                )
                self.assertEqual(mo_x_dip_ints._matrices[0].shape, (2, 2))
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_x_dip_ints._matrices[0]), [[0.0, 0.0], [0.0, 0.0]], decimal=4
                )

            with self.subTest("y axis"):
                mo_y_dip_ints = dipole.get_property("DipoleMomentY").get_electronic_integral(
                    ElectronicBasis.MO, 1
                )
                self.assertEqual(mo_y_dip_ints._matrices[0].shape, (2, 2))
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_y_dip_ints._matrices[0]), [[0.0, 0.0], [0.0, 0.0]], decimal=4
                )

            with self.subTest("z axis"):
                mo_z_dip_ints = dipole.get_property("DipoleMomentZ").get_electronic_integral(
                    ElectronicBasis.MO, 1
                )
                self.assertEqual(mo_z_dip_ints._matrices[0].shape, (2, 2))
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_z_dip_ints._matrices[0]),
                    [[0.6945, 0.9278], [0.9278, 0.6945]],
                    decimal=4,
                )

            with self.subTest("nuclear dipole moment"):
                np.testing.assert_array_almost_equal(
                    np.absolute(dipole.nuclear_dipole_moment), [0.0, 0.0, 1.3889], decimal=4
                )
