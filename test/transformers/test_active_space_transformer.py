# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ActiveSpaceTransformer."""

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, idata, unpack
import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.transformers import ActiveSpaceTransformer


@ddt
class TestActiveSpaceTransformer(QiskitNatureTestCase):
    """ActiveSpaceTransformer tests."""

    def test_full_active_space(self):
        """Test that transformer has no effect when all orbitals are active."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_sto3g.hdf5', 'transformers'))
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_orbitals=2)
        q_molecule_reduced = trafo.transform(q_molecule)

        assert np.allclose(q_molecule_reduced.mo_onee_ints, q_molecule.mo_onee_ints)
        assert np.allclose(q_molecule_reduced.mo_eri_ints, q_molecule.mo_eri_ints)
        assert np.isclose(q_molecule_reduced.energy_shift['inactive_energy'], 0.0)

    def test_minimal_active_space(self):
        """Test a minimal active space manually."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_orbitals=2)
        q_molecule_reduced = trafo.transform(q_molecule)

        expected_mo_onee_ints = np.asarray([[-1.24943841, 0.0], [0.0, -0.547816138]])
        expected_mo_eri_ints = np.asarray([[[[0.652098466, 0.0], [0.0, 0.433536565]],
                                            [[0.0, 0.0794483182], [0.0794483182, 0.0]]],
                                           [[[0.0, 0.0794483182], [0.0794483182, 0.0]],
                                            [[0.433536565, 0.0], [0.0, 0.385524695]]]])

        assert np.allclose(q_molecule_reduced.mo_onee_ints, expected_mo_onee_ints)
        assert np.allclose(q_molecule_reduced.mo_eri_ints, expected_mo_eri_ints)
        assert np.isclose(q_molecule_reduced.energy_shift['inactive_energy'], 0.0)

    def test_unpaired_electron_active_space(self):
        """Test an active space with an unpaired electron."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('BeH_sto3g.hdf5', 'transformers'))
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=3, num_orbitals=3, num_alpha=2)
        q_molecule_reduced = trafo.transform(q_molecule)

        expected_mo_onee_ints = np.asarray([[-1.30228816, 0.03573328, 0.0],
                                            [0.03573328, -0.86652349, 0.0],
                                            [0.0, 0.0, -0.84868407]])

        expected_mo_eri_ints = np.asarray([[[[0.57237421, -0.05593597, 0.0],
                                             [-0.05593597, 0.30428426, 0.0],
                                             [0.0, 0.0, 0.36650821]],
                                            [[-0.05593597, 0.01937529, 0.0],
                                             [0.01937529, 0.02020237, 0.0],
                                             [0.0, 0.0, 0.01405676]],
                                            [[0.0, 0.0, 0.03600701],
                                             [0.0, 0.0, 0.028244],
                                             [0.03600701, 0.028244, 0.0]]],
                                           [[[-0.05593597, 0.01937529, 0.0],
                                             [0.01937529, 0.02020237, 0.0],
                                             [0.0, 0.0, 0.01405676]],
                                            [[0.30428426, 0.02020237, 0.0],
                                             [0.02020237, 0.48162669, 0.0],
                                             [0.0, 0.0, 0.40269913]],
                                            [[0.0, 0.0, 0.028244],
                                             [0.0, 0.0, 0.0564951],
                                             [0.028244, 0.0564951, 0.0]]],
                                           [[[0.0, 0.0, 0.03600701],
                                             [0.0, 0.0, 0.028244],
                                             [0.03600701, 0.028244, 0.0]],
                                            [[0.0, 0.0, 0.028244],
                                             [0.0, 0.0, 0.0564951],
                                             [0.028244, 0.0564951, 0.0]],
                                            [[0.36650821, 0.01405676, 0.0],
                                             [0.01405676, 0.40269913, 0.0],
                                             [0.0, 0.0, 0.44985904]]]])

        assert np.allclose(q_molecule_reduced.mo_onee_ints, expected_mo_onee_ints)
        assert np.allclose(q_molecule_reduced.mo_eri_ints, expected_mo_eri_ints)
        assert np.isclose(q_molecule_reduced.energy_shift['inactive_energy'], -14.2538029231)

    def test_arbitrary_active_orbitals(self):
        """Test manual selection of active orbital indices."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_orbitals=2, active_orbitals=[0, 2])
        q_molecule_reduced = trafo.transform(q_molecule)

        expected_mo_onee_ints = np.asarray([[-1.24943841, -0.16790838], [-0.16790838, -0.18307469]])
        expected_mo_eri_ints = np.asarray([[[[0.65209847, 0.16790822], [0.16790822, 0.53250905]],
                                            [[0.16790822, 0.10962908], [0.10962908, 0.11981429]]],
                                           [[[0.16790822, 0.10962908], [0.10962908, 0.11981429]],
                                            [[0.53250905, 0.11981429], [0.11981429, 0.46345617]]]])

        assert np.allclose(q_molecule_reduced.mo_onee_ints, expected_mo_onee_ints)
        assert np.allclose(q_molecule_reduced.mo_eri_ints, expected_mo_eri_ints)
        assert np.isclose(q_molecule_reduced.energy_shift['inactive_energy'], 0.0)

    @idata([
        [2, 3, None, "More active orbitals requested than available in total."],
        [3, 2, None, "More active electrons requested than available in total."],
        [1, 2, None, "The number of inactive electrons may not be odd."],
        [2, 2, [0, 1, 2], "The number of active orbitals do not match."],
        [2, 2, [1, 2], "The number of active electrons do not match."],
    ])
    @unpack
    def test_error_raising(self, num_electrons, num_orbitals, active_orbitals, message):
        """Test errors are being raised in certain scenarios."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_sto3g.hdf5', 'transformers'))
        q_molecule = driver.run()

        with self.assertRaises(QiskitNatureError, msg=message):
            ActiveSpaceTransformer(num_electrons=num_electrons,
                                   num_orbitals=num_orbitals,
                                   active_orbitals=active_orbitals).transform(q_molecule)

    def test_active_space_for_q_molecule_v2(self):
        """Test based on QMolecule v2 (mo_occ not available)."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_sto3g_v2.hdf5', 'transformers'))
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_orbitals=2)
        q_molecule_reduced = trafo.transform(q_molecule)

        assert np.allclose(q_molecule_reduced.mo_onee_ints, q_molecule.mo_onee_ints)
        assert np.allclose(q_molecule_reduced.mo_eri_ints, q_molecule.mo_eri_ints)
        assert np.isclose(q_molecule_reduced.energy_shift['inactive_energy'], 0.0)


if __name__ == '__main__':
    unittest.main()
