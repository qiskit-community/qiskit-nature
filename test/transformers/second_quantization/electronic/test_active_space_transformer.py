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
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicDriverResult,
    ElectronicEnergy,
    TotalDipoleMoment,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.dipole_moment import DipoleMoment
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer


# With Python 3.6 this false positive is being raised for the ElectronicDriverResult
# pylint: disable=abstract-class-instantiated
@ddt
class TestActiveSpaceTransformer(QiskitNatureTestCase):
    """ActiveSpaceTransformer tests."""

    def assertDriverResult(self, driver_result, expected, dict_key="ActiveSpaceTransformer"):
        """Asserts that the two `DriverResult` object's relevant fields are equivalent."""
        electronic_energy = driver_result.get_property("ElectronicEnergy")
        electronic_energy_exp = expected.get_property("ElectronicEnergy")
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(
                electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin(),
                electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 1).to_spin(),
            )
        with self.subTest("MO 2-electron integrals"):
            np.testing.assert_array_almost_equal(
                electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin(),
                electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 2).to_spin(),
            )
        with self.subTest("Inactive energy"):
            self.assertAlmostEqual(
                electronic_energy._shift[dict_key],
                electronic_energy_exp._shift["ActiveSpaceTransformer"],
            )

        for dipole, dipole_exp in zip(
            iter(driver_result.get_property("TotalDipoleMoment")),
            iter(expected.get_property("TotalDipoleMoment")),
        ):
            with self.subTest(f"MO 1-electron {dipole._axis} dipole integrals"):
                np.testing.assert_array_almost_equal(
                    dipole.get_electronic_integral(ElectronicBasis.MO, 1).to_spin(),
                    dipole_exp.get_electronic_integral(ElectronicBasis.MO, 1).to_spin(),
                )
            with self.subTest(f"{dipole._axis} dipole energy shift"):
                self.assertAlmostEqual(
                    dipole._shift[dict_key],
                    dipole_exp._shift["ActiveSpaceTransformer"],
                )

    @idata(
        [
            {"num_electrons": 2, "num_molecular_orbitals": 2},
        ]
    )
    def test_full_active_space(self, kwargs):
        """test that transformer has no effect when all orbitals are active."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicDriverResult.from_legacy_driver_result(q_molecule)

        driver_result.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"] = 0.0
        for prop in iter(driver_result.get_property("TotalDipoleMoment")):
            prop._shift["ActiveSpaceTransformer"] = 0.0

        trafo = ActiveSpaceTransformer(**kwargs)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(driver_result_reduced, driver_result)

    def test_minimal_active_space(self):
        """Test a minimal active space manually."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_631g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicDriverResult.from_legacy_driver_result(q_molecule)

        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicDriverResult()
        expected.add_property(
            ElectronicEnergy(
                [
                    OneBodyElectronicIntegrals(
                        ElectronicBasis.MO,
                        (np.asarray([[-1.24943841, 0.0], [0.0, -0.547816138]]), None),
                    ),
                    TwoBodyElectronicIntegrals(
                        ElectronicBasis.MO,
                        (
                            np.asarray(
                                [
                                    [
                                        [[0.652098466, 0.0], [0.0, 0.433536565]],
                                        [[0.0, 0.0794483182], [0.0794483182, 0.0]],
                                    ],
                                    [
                                        [[0.0, 0.0794483182], [0.0794483182, 0.0]],
                                        [[0.433536565, 0.0], [0.0, 0.385524695]],
                                    ],
                                ]
                            ),
                            None,
                            None,
                            None,
                        ),
                    ),
                ],
                energy_shift={"ActiveSpaceTransformer": 0.0},
            )
        )
        expected.add_property(
            TotalDipoleMoment(
                [
                    DipoleMoment(
                        "x",
                        [OneBodyElectronicIntegrals(ElectronicBasis.MO, (np.zeros((2, 2)), None))],
                        shift={"ActiveSpaceTransformer": 0.0},
                    ),
                    DipoleMoment(
                        "y",
                        [OneBodyElectronicIntegrals(ElectronicBasis.MO, (np.zeros((2, 2)), None))],
                        shift={"ActiveSpaceTransformer": 0.0},
                    ),
                    DipoleMoment(
                        "z",
                        [
                            OneBodyElectronicIntegrals(
                                ElectronicBasis.MO,
                                (
                                    np.asarray(
                                        [[0.69447435, -1.01418298], [-1.01418298, 0.69447435]]
                                    ),
                                    None,
                                ),
                            )
                        ],
                        shift={"ActiveSpaceTransformer": 0.0},
                    ),
                ]
            )
        )

        self.assertDriverResult(driver_result_reduced, expected)

    def test_unpaired_electron_active_space(self):
        """Test an active space with an unpaired electron."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "BeH_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicDriverResult.from_legacy_driver_result(q_molecule)

        trafo = ActiveSpaceTransformer(num_electrons=(2, 1), num_molecular_orbitals=3)
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicDriverResult.from_legacy_driver_result(
            HDF5Driver(
                hdf5_input=self.get_resource_path(
                    "BeH_sto3g_reduced.hdf5", "transformers/second_quantization/electronic"
                )
            ).run()
        )

        self.assertDriverResult(driver_result_reduced, expected)

    def test_arbitrary_active_orbitals(self):
        """Test manual selection of active orbital indices."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_631g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicDriverResult.from_legacy_driver_result(q_molecule)

        trafo = ActiveSpaceTransformer(
            num_electrons=2, num_molecular_orbitals=2, active_orbitals=[0, 2]
        )
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicDriverResult()
        expected.add_property(
            ElectronicEnergy(
                [
                    OneBodyElectronicIntegrals(
                        ElectronicBasis.MO,
                        (
                            np.asarray([[-1.24943841, -0.16790838], [-0.16790838, -0.18307469]]),
                            None,
                        ),
                    ),
                    TwoBodyElectronicIntegrals(
                        ElectronicBasis.MO,
                        (
                            np.asarray(
                                [
                                    [
                                        [[0.65209847, 0.16790822], [0.16790822, 0.53250905]],
                                        [[0.16790822, 0.10962908], [0.10962908, 0.11981429]],
                                    ],
                                    [
                                        [[0.16790822, 0.10962908], [0.10962908, 0.11981429]],
                                        [[0.53250905, 0.11981429], [0.11981429, 0.46345617]],
                                    ],
                                ]
                            ),
                            None,
                            None,
                            None,
                        ),
                    ),
                ],
                energy_shift={"ActiveSpaceTransformer": 0.0},
            )
        )
        expected.add_property(
            TotalDipoleMoment(
                [
                    DipoleMoment(
                        "x",
                        [OneBodyElectronicIntegrals(ElectronicBasis.MO, (np.zeros((2, 2)), None))],
                        shift={"ActiveSpaceTransformer": 0.0},
                    ),
                    DipoleMoment(
                        "y",
                        [OneBodyElectronicIntegrals(ElectronicBasis.MO, (np.zeros((2, 2)), None))],
                        shift={"ActiveSpaceTransformer": 0.0},
                    ),
                    DipoleMoment(
                        "z",
                        [
                            OneBodyElectronicIntegrals(
                                ElectronicBasis.MO,
                                (np.asarray([[0.69447435, 0.0], [0.0, 0.69447435]]), None),
                            )
                        ],
                        shift={"ActiveSpaceTransformer": 0.0},
                    ),
                ]
            )
        )
        self.assertDriverResult(driver_result_reduced, expected)

    @idata(
        [
            [2, 3, None, "More active orbitals requested than available in total."],
            [4, 2, None, "More active electrons requested than available in total."],
            [(1, 0), 2, None, "The number of inactive electrons may not be odd."],
            [2, 2, [0, 1, 2], "The number of active orbitals do not match."],
            [2, 2, [1, 2], "The number of active electrons do not match."],
            [1, 2, None, "The number of active electrons must be even when not a tuple."],
            [-2, 2, None, "The number of active electrons must not be negative."],
        ]
    )
    @unpack
    def test_error_raising(self, num_electrons, num_molecular_orbitals, active_orbitals, message):
        """Test errors are being raised in certain scenarios."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicDriverResult.from_legacy_driver_result(q_molecule)

        with self.assertRaises(QiskitNatureError, msg=message):
            ActiveSpaceTransformer(
                num_electrons=num_electrons,
                num_molecular_orbitals=num_molecular_orbitals,
                active_orbitals=active_orbitals,
            ).transform(driver_result)

    def test_active_space_for_q_molecule_v2(self):
        """Test based on QMolecule v2 (mo_occ not available)."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_sto3g_v2.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicDriverResult.from_legacy_driver_result(q_molecule)

        driver_result.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"] = 0.0
        for prop in iter(driver_result.get_property("TotalDipoleMoment")):
            prop._shift["ActiveSpaceTransformer"] = 0.0

        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(driver_result_reduced, driver_result)


if __name__ == "__main__":
    unittest.main()
