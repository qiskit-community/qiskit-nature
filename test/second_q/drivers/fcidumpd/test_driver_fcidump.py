# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver FCIDump """

from typing import cast

import unittest
from abc import ABC, abstractmethod
from test import QiskitNatureTestCase
import numpy as np
from qiskit_nature.second_q.drivers import FCIDumpDriver
from qiskit_nature.second_q.operator_factories.electronic import (
    ElectronicEnergy,
    ParticleNumber,
)
from qiskit_nature.second_q.operator_factories.electronic.bases import ElectronicBasis


class BaseTestDriverFCIDump(ABC):
    """FCIDump Driver base test class.

    In contrast to the other driver tests this one does *not* derive from TestDriver because the
    interface is fundamentally different.
    """

    def __init__(self):
        self.log = None
        self.driver_result = None
        self.nuclear_repulsion_energy = None
        self.num_molecular_orbitals = None
        self.num_alpha = None
        self.num_beta = None
        self.mo_onee = None
        self.mo_onee_b = None
        self.mo_eri = None
        self.mo_eri_ba = None
        self.mo_eri_bb = None

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

        with self.subTest("inactive energy"):
            self.log.debug("inactive energy: %s", electronic_energy.nuclear_repulsion_energy)
            self.assertAlmostEqual(
                electronic_energy.nuclear_repulsion_energy,
                self.nuclear_repulsion_energy,
                places=3,
            )

        mo_onee_ints = electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1)
        with self.subTest("1-body alpha"):
            self.log.debug("MO one electron integrals are %s", mo_onee_ints)
            self.assertEqual(mo_onee_ints._matrices[0].shape, self.mo_onee.shape)
            np.testing.assert_array_almost_equal(
                np.absolute(mo_onee_ints._matrices[0]),
                np.absolute(self.mo_onee),
                decimal=4,
            )

        if self.mo_onee_b is not None:
            with self.subTest("1-body beta"):
                self.assertEqual(mo_onee_ints._matrices[1].shape, self.mo_onee_b.shape)
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_onee_ints._matrices[1]),
                    np.absolute(self.mo_onee_b),
                    decimal=4,
                )

        mo_eri_ints = electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
        with self.subTest("2-body alpha-alpha"):
            self.log.debug("MO two electron integrals %s", mo_eri_ints)
            self.assertEqual(mo_eri_ints._matrices[0].shape, self.mo_eri.shape)
            np.testing.assert_array_almost_equal(
                np.absolute(mo_eri_ints._matrices[0]), np.absolute(self.mo_eri), decimal=4
            )

        if self.mo_eri_ba is not None:
            with self.subTest("2-body beta-alpha"):
                self.assertEqual(mo_eri_ints._matrices[1].shape, self.mo_eri_ba.shape)
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_eri_ints._matrices[1]), np.absolute(self.mo_eri_ba), decimal=4
                )

        if self.mo_eri_bb is not None:
            with self.subTest("2-body beta-beta"):
                self.assertEqual(mo_eri_ints._matrices[2].shape, self.mo_eri_bb.shape)
                np.testing.assert_array_almost_equal(
                    np.absolute(mo_eri_ints._matrices[2]), np.absolute(self.mo_eri_bb), decimal=4
                )

    def test_driver_result_particle_number(self):
        """Test the ParticleNumber property."""
        particle_number = cast(ParticleNumber, self.driver_result.get_property(ParticleNumber))

        with self.subTest("orbital number"):
            self.log.debug("Number of orbitals is %s", particle_number.num_spin_orbitals)
            self.assertEqual(particle_number.num_spin_orbitals, self.num_molecular_orbitals * 2)

        with self.subTest("alpha electron number"):
            self.log.debug("Number of alpha electrons is %s", particle_number.num_alpha)
            self.assertEqual(particle_number.num_alpha, self.num_alpha)

        with self.subTest("beta electron number"):
            self.log.debug("Number of beta electrons is %s", particle_number.num_beta)
            self.assertEqual(particle_number.num_beta, self.num_beta)


class TestDriverFCIDumpH2(QiskitNatureTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 0.7199
        self.num_molecular_orbitals = 2
        self.num_alpha = 1
        self.num_beta = 1
        self.mo_onee = np.array([[1.2563, 0.0], [0.0, 0.4719]])
        self.mo_onee_b = None
        self.mo_eri = np.array(
            [
                [[[0.6757, 0.0], [0.0, 0.6646]], [[0.0, 0.1809], [0.1809, 0.0]]],
                [[[0.0, 0.1809], [0.1809, 0.0]], [[0.6646, 0.0], [0.0, 0.6986]]],
            ]
        )
        self.mo_eri_ba = None
        self.mo_eri_bb = None
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_h2.fcidump", "second_q/drivers/fcidumpd")
        )
        self.driver_result = driver.run()


class TestDriverFCIDumpLiH(QiskitNatureTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 0.9924
        self.num_molecular_orbitals = 6
        self.num_alpha = 2
        self.num_beta = 2
        loaded = np.load(
            self.get_resource_path("test_driver_fcidump_lih.npz", "second_q/drivers/fcidumpd")
        )
        self.mo_onee = loaded["mo_onee"]
        self.mo_onee_b = None
        self.mo_eri = loaded["mo_eri"]
        self.mo_eri_ba = None
        self.mo_eri_bb = None
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_lih.fcidump", "second_q/drivers/fcidumpd")
        )
        self.driver_result = driver.run()


class TestDriverFCIDumpOH(QiskitNatureTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 11.3412
        self.num_molecular_orbitals = 6
        self.num_alpha = 5
        self.num_beta = 4
        loaded = np.load(
            self.get_resource_path("test_driver_fcidump_oh.npz", "second_q/drivers/fcidumpd")
        )
        self.mo_onee = loaded["mo_onee"]
        self.mo_onee_b = loaded["mo_onee_b"]
        self.mo_eri = loaded["mo_eri"]
        self.mo_eri_ba = loaded["mo_eri_ba"]
        self.mo_eri_bb = loaded["mo_eri_bb"]
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_oh.fcidump", "second_q/drivers/fcidumpd")
        )
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
