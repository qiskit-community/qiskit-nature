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

"""Tests for the FreezeCoreTransformer."""

import unittest

from test.transformers.second_quantization.test_active_space_transformer import (
    TestActiveSpaceTransformer,
)

from ddt import ddt, idata

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.transformers.second_quantization import FreezeCoreTransformer


@ddt
class TestFreezeCoreTransformer(TestActiveSpaceTransformer):
    """FreezeCoreTransformer tests."""

    @idata(
        [
            {"freeze_core": True},
        ]
    )
    def test_full_active_space(self, kwargs):
        """Test that transformer has no effect when all orbitals are active."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_sto3g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        # The references which we compare too were produced by the `ActiveSpaceTransformer` and,
        # thus, the key here needs to stay the same as in that test case.
        q_molecule.energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.x_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.y_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.z_dip_energy_shift["ActiveSpaceTransformer"] = 0.0

        trafo = FreezeCoreTransformer(**kwargs)
        q_molecule_reduced = trafo.transform(q_molecule)

        self.assertQMolecule(q_molecule_reduced, q_molecule, dict_key="FreezeCoreTransformer")

    def test_freeze_core(self):
        """Test the `freeze_core` convenience argument."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("LiH_sto3g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        trafo = FreezeCoreTransformer(freeze_core=True)
        q_molecule_reduced = trafo.transform(q_molecule)

        expected = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g_reduced.hdf5", "transformers/second_quantization"
            )
        ).run()

        self.assertQMolecule(q_molecule_reduced, expected, dict_key="FreezeCoreTransformer")

    def test_freeze_core_with_remove_orbitals(self):
        """Test the `freeze_core` convenience argument in combination with `remove_orbitals`."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("BeH_sto3g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        trafo = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[4, 5])
        q_molecule_reduced = trafo.transform(q_molecule)

        expected = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "BeH_sto3g_reduced.hdf5", "transformers/second_quantization"
            )
        ).run()

        self.assertQMolecule(q_molecule_reduced, expected, dict_key="FreezeCoreTransformer")


if __name__ == "__main__":
    unittest.main()
