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

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators.tensor_ordering import _chem_to_phys
from qiskit_nature.second_q.problems import ElectronicStructureProblem


class TestDriver(ABC):
    """Common driver tests. For H2 @ 0.735, sto3g"""

    MOLECULE = MoleculeInfo(
        symbols=["H", "H"],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.735)],
        multiplicity=1,
        charge=0,
        units=DistanceUnit.ANGSTROM,
    )

    def __init__(self):
        self.log = None
        self.driver_result: ElectronicStructureProblem = None

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
        electronic_energy = cast(ElectronicEnergy, self.driver_result.hamiltonian)

        with self.subTest("reference energy"):
            self.log.debug("HF energy: %s", self.driver_result.reference_energy)
            self.assertAlmostEqual(self.driver_result.reference_energy, -1.117, places=3)

        with self.subTest("nuclear repulsion energy"):
            self.log.debug(
                "Nuclear repulsion energy: %s", self.driver_result.nuclear_repulsion_energy
            )
            self.assertAlmostEqual(self.driver_result.nuclear_repulsion_energy, 0.72, places=2)

        with self.subTest("1-body integrals"):
            mo_onee_ints = electronic_energy.electronic_integrals.alpha["+-"]
            self.log.debug("MO one electron integrals %s", mo_onee_ints)
            self.assertEqual(mo_onee_ints.shape, (2, 2))
            np.testing.assert_array_almost_equal(
                np.absolute(mo_onee_ints),
                [[1.2563, 0.0], [0.0, 0.4719]],
                decimal=4,
            )

        with self.subTest("2-body integrals"):
            mo_eri_ints = electronic_energy.electronic_integrals.alpha["++--"]
            self.log.debug("MO two electron integrals %s", mo_eri_ints)
            self.assertEqual(mo_eri_ints.shape, (2, 2, 2, 2))
            np.testing.assert_array_almost_equal(
                np.absolute(mo_eri_ints),
                _chem_to_phys(
                    np.asarray(
                        [
                            [[[0.6757, 0.0], [0.0, 0.6646]], [[0.0, 0.1809], [0.1809, 0.0]]],
                            [[[0.0, 0.1809], [0.1809, 0.0]], [[0.6646, 0.0], [0.0, 0.6986]]],
                        ]
                    )
                ),
                decimal=4,
            )

    def test_driver_result_system_size(self):
        """Test the system size problem attributes."""

        with self.subTest("orbital number"):
            self.log.debug("Number of orbitals is %s", self.driver_result.num_spatial_orbitals)
            self.assertEqual(self.driver_result.num_spatial_orbitals, 2)

        with self.subTest("alpha electron number"):
            self.log.debug("Number of alpha electrons is %s", self.driver_result.num_alpha)
            self.assertEqual(self.driver_result.num_alpha, 1)

        with self.subTest("beta electron number"):
            self.log.debug("Number of beta electrons is %s", self.driver_result.num_beta)
            self.assertEqual(self.driver_result.num_beta, 1)

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
            self.log.debug("num atoms %s", len(molecule.symbols))
            self.assertEqual(len(molecule.symbols), 2)

        with self.subTest("atoms"):
            self.log.debug("atom symbol %s", molecule.symbols)
            self.assertSequenceEqual(molecule.symbols, ["H", "H"])

        with self.subTest("coordinates"):
            coords = np.asarray(molecule.coords)
            self.log.debug("atom xyz %s", coords)
            np.testing.assert_array_almost_equal(
                coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3889]], decimal=4
            )

    def test_driver_result_electronic_dipole(self):
        """Test the ElectronicDipoleMoment property."""
        dipole = self.driver_result.properties.electronic_dipole_moment

        self.log.debug("has dipole integrals %s", dipole is not None)
        if dipole is not None:
            with self.subTest("x axis"):
                mo_x_dip_ints = dipole.x_dipole.alpha["+-"]
                self.assertEqual(mo_x_dip_ints.shape, (2, 2))
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_x_dip_ints), [[0.0, 0.0], [0.0, 0.0]], decimal=4
                )

            with self.subTest("y axis"):
                mo_y_dip_ints = dipole.y_dipole.alpha["+-"]
                self.assertEqual(mo_y_dip_ints.shape, (2, 2))
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_y_dip_ints), [[0.0, 0.0], [0.0, 0.0]], decimal=4
                )

            with self.subTest("z axis"):
                mo_z_dip_ints = dipole.z_dipole.alpha["+-"]
                self.assertEqual(mo_z_dip_ints.shape, (2, 2))
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_z_dip_ints),
                    [[0.6945, 0.9278], [0.9278, 0.6945]],
                    decimal=4,
                )

            with self.subTest("nuclear dipole moment"):
                np.testing.assert_array_almost_equal(
                    np.absolute(dipole.nuclear_dipole_moment), [0.0, 0.0, 1.3889], decimal=4
                )
