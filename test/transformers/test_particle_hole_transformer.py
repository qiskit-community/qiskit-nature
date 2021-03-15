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

import numpy as np

from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.transformers import ParticleHoleTransformer


class TestParticleHoleTransformer(QiskitNatureTestCase):
    """ParticleHoleTransformer tests."""

    def test_particle_hole_energy_shift_and_ints(self):
        """ particle hole test """

        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_sto3g.hdf5', 'transformers'))
        q_molecule = driver.run()

        _ = q_molecule.num_alpha

        trafo = ParticleHoleTransformer(num_electrons=2, num_orbitals=2, num_alpha=1)
        q_molecule_transformed = trafo.transform(q_molecule)

        ph_shift = q_molecule_transformed.energy_shift['ParticleHoleTransformer']

        # ph_shift should be the electronic part of the hartree fock energy

        self.assertAlmostEqual(-ph_shift,
                               q_molecule.hf_energy - q_molecule.nuclear_repulsion_energy)

        expected_h1 = np.load(self.get_resource_path('ph_onee_ints_test.npy', 'transformers'))
        expected_h2 = np.load(self.get_resource_path('ph_eri_ints_test.npy', 'transformers'))

        assert np.allclose(expected_h1, q_molecule_transformed.one_body_integrals)

        # TODO fix calculation of two body integrals reverse the twoe_to_spin
        # or add setter in algorithms
        assert np.allclose(expected_h2, q_molecule_transformed.two_body_integrals)


if __name__ == '__main__':
    unittest.main()
