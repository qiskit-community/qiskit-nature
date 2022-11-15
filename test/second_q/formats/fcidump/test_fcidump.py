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

""" Test FCIDump """

import unittest
from abc import ABC, abstractmethod
from test import QiskitNatureTestCase
import numpy as np
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.operators.tensor_ordering import _chem_to_phys


class BaseTestFCIDump(ABC):
    """FCIDump base test class."""

    def __init__(self):
        self.log = None
        self.problem = None
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

    def test_electronic_energy(self):
        """Test the ElectronicEnergy property."""

        electronic_energy = self.problem.hamiltonian

        with self.subTest("inactive energy"):
            self.log.debug("inactive energy: %s", electronic_energy.nuclear_repulsion_energy)
            self.assertAlmostEqual(
                electronic_energy.nuclear_repulsion_energy,
                self.nuclear_repulsion_energy,
                places=3,
            )

        with self.subTest("1-body alpha"):
            alpha_1body = electronic_energy.electronic_integrals.alpha["+-"]
            self.log.debug("MO one electron alpha integrals are %s", alpha_1body)
            self.assertEqual(alpha_1body.shape, self.mo_onee.shape)
            np.testing.assert_array_almost_equal(
                np.absolute(alpha_1body), np.absolute(self.mo_onee), decimal=4
            )

        if self.mo_onee_b is not None:
            with self.subTest("1-body beta"):
                beta_1body = electronic_energy.electronic_integrals.beta["+-"]
                self.log.debug("MO one electron beta integrals are %s", beta_1body)
                self.assertEqual(beta_1body.shape, self.mo_onee_b.shape)
                np.testing.assert_array_almost_equal(
                    np.absolute(beta_1body), np.absolute(self.mo_onee_b), decimal=4
                )

        with self.subTest("2-body alpha-alpha"):
            alpha_2body = electronic_energy.electronic_integrals.alpha["++--"]
            self.log.debug("MO two electron alpha-alpha integrals are %s", alpha_2body)
            self.assertEqual(alpha_2body.shape, self.mo_eri.shape)
            np.testing.assert_array_almost_equal(
                np.absolute(alpha_2body), np.absolute(self.mo_eri), decimal=4
            )

        if self.mo_eri_ba is not None:
            with self.subTest("2-body beta-alpha"):
                beta_alpha_2body = electronic_energy.electronic_integrals.beta_alpha["++--"]
                self.log.debug("MO two electron beta-alpha integrals are %s", beta_alpha_2body)
                self.assertEqual(beta_alpha_2body.shape, self.mo_eri_ba.shape)
                np.testing.assert_array_almost_equal(
                    np.absolute(beta_alpha_2body), np.absolute(self.mo_eri_ba), decimal=4
                )

        if self.mo_eri_bb is not None:
            with self.subTest("2-body beta-beta"):
                beta_2body = electronic_energy.electronic_integrals.beta["++--"]
                self.log.debug("MO two electron beta-alpha integrals are %s", beta_2body)
                self.assertEqual(beta_2body.shape, self.mo_eri_bb.shape)
                np.testing.assert_array_almost_equal(
                    np.absolute(beta_2body), np.absolute(self.mo_eri_bb), decimal=4
                )

    def test_system_size(self):
        """Test the system size problem attributes."""

        with self.subTest("orbital number"):
            self.log.debug("Number of orbitals is %s", self.problem.num_spatial_orbitals)
            self.assertEqual(self.problem.num_spatial_orbitals, self.num_molecular_orbitals)

        with self.subTest("alpha electron number"):
            self.log.debug("Number of alpha electrons is %s", self.problem.num_alpha)
            self.assertEqual(self.problem.num_alpha, self.num_alpha)

        with self.subTest("beta electron number"):
            self.log.debug("Number of beta electrons is %s", self.problem.num_beta)
            self.assertEqual(self.problem.num_beta, self.num_beta)


class TestFCIDumpH2(QiskitNatureTestCase, BaseTestFCIDump):
    """RHF H2 FCIDump tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 0.7199
        self.num_molecular_orbitals = 2
        self.num_alpha = 1
        self.num_beta = 1
        self.mo_onee = np.array([[1.2563, 0.0], [0.0, 0.4719]])
        self.mo_onee_b = None
        self.mo_eri = _chem_to_phys(
            np.array(
                [
                    [[[0.6757, 0.0], [0.0, 0.6646]], [[0.0, 0.1809], [0.1809, 0.0]]],
                    [[[0.0, 0.1809], [0.1809, 0.0]], [[0.6646, 0.0], [0.0, 0.6986]]],
                ]
            )
        )
        self.mo_eri_ba = None
        self.mo_eri_bb = None
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_h2.fcidump", "second_q/formats/fcidump")
        )
        self.problem = fcidump_to_problem(fcidump)


class TestFCIDumpLiH(QiskitNatureTestCase, BaseTestFCIDump):
    """RHF LiH FCIDump tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 0.9924
        self.num_molecular_orbitals = 6
        self.num_alpha = 2
        self.num_beta = 2
        loaded = np.load(self.get_resource_path("test_fcidump_lih.npz", "second_q/formats/fcidump"))
        self.mo_onee = loaded["mo_onee"]
        self.mo_onee_b = None
        self.mo_eri = _chem_to_phys(loaded["mo_eri"])
        self.mo_eri_ba = None
        self.mo_eri_bb = None
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_lih.fcidump", "second_q/formats/fcidump")
        )
        self.problem = fcidump_to_problem(fcidump)


class TestFCIDumpOH(QiskitNatureTestCase, BaseTestFCIDump):
    """UHF OH FCIDump tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 11.3412
        self.num_molecular_orbitals = 6
        self.num_alpha = 5
        self.num_beta = 4
        loaded = np.load(self.get_resource_path("test_fcidump_oh.npz", "second_q/formats/fcidump"))
        self.mo_onee = loaded["mo_onee"]
        self.mo_onee_b = loaded["mo_onee_b"]
        self.mo_eri = _chem_to_phys(loaded["mo_eri"])
        self.mo_eri_ba = _chem_to_phys(loaded["mo_eri_ba"])
        self.mo_eri_bb = _chem_to_phys(loaded["mo_eri_bb"])
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_oh.fcidump", "second_q/formats/fcidump")
        )
        self.problem = fcidump_to_problem(fcidump)


if __name__ == "__main__":
    unittest.main()
