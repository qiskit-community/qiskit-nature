# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.properties.dipole_moment import DipoleMoment
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


@ddt
class TestActiveSpaceTransformer(QiskitNatureTestCase):
    """ActiveSpaceTransformer tests."""

    def assertDriverResult(self, driver_result, expected, dict_key="ActiveSpaceTransformer"):
        """Asserts that the two `DriverResult` object's relevant fields are equivalent."""
        electronic_energy = driver_result.hamiltonian
        electronic_energy_exp = expected.hamiltonian
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(
                np.abs(electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()),
                np.abs(
                    electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()
                ),
            )
        with self.subTest("MO 2-electron integrals"):
            np.testing.assert_array_almost_equal(
                np.abs(electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()),
                np.abs(
                    electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 2).to_spin()
                ),
            )
        with self.subTest("Inactive energy"):
            self.assertAlmostEqual(
                electronic_energy._shift[dict_key],
                electronic_energy_exp._shift["ActiveSpaceTransformer"],
            )

        if expected.properties.electronic_dipole_moment is not None:
            for dipole, dipole_exp in zip(
                driver_result.properties.electronic_dipole_moment._dipole_axes.values(),
                expected.properties.electronic_dipole_moment._dipole_axes.values(),
            ):
                with self.subTest(f"MO 1-electron {dipole._axis} dipole integrals"):
                    np.testing.assert_array_almost_equal(
                        np.abs(dipole.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()),
                        np.abs(dipole_exp.get_electronic_integral(ElectronicBasis.MO, 1).to_spin()),
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
        driver = PySCFDriver()
        driver_result = driver.run()

        driver_result.hamiltonian._shift["ActiveSpaceTransformer"] = 0.0
        for prop in driver_result.properties.electronic_dipole_moment._dipole_axes.values():
            prop._shift["ActiveSpaceTransformer"] = 0.0

        trafo = ActiveSpaceTransformer(**kwargs)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(driver_result_reduced, driver_result)

    def test_minimal_active_space(self):
        """Test a minimal active space manually."""
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicStructureProblem(
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
        expected.properties.electronic_dipole_moment = ElectronicDipoleMoment(
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
                                np.asarray([[0.69447435, -1.01418298], [-1.01418298, 0.69447435]]),
                                None,
                            ),
                        )
                    ],
                    shift={"ActiveSpaceTransformer": 0.0},
                ),
            ]
        )

        self.assertDriverResult(driver_result_reduced, expected)

    def test_unpaired_electron_active_space(self):
        """Test an active space with an unpaired electron."""
        driver = PySCFDriver(atom="Be 0 0 0; H 0 0 1.3", basis="sto3g", spin=1)
        driver_result = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=(2, 1), num_molecular_orbitals=3)
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicStructureProblem(
            ElectronicEnergy(
                [
                    OneBodyElectronicIntegrals(
                        ElectronicBasis.MO,
                        (
                            np.asarray(
                                [
                                    [-1.30228816, 0.03573328, 0.0],
                                    [0.03573328, -0.86652349, 0.0],
                                    [0.0, 0.0, -0.84868407],
                                ]
                            ),
                            None,
                        ),
                    ),
                    TwoBodyElectronicIntegrals(
                        ElectronicBasis.MO,
                        (
                            np.asarray(
                                [
                                    [
                                        [
                                            [0.5723742128971601, -0.055935974537879886, 0.0],
                                            [-0.0559359745378799, 0.3042842626235253, 0.0],
                                            [0.0, 0.0, 0.3665082095336567],
                                        ],
                                        [
                                            [-0.05593597453787989, 0.01937528764526897, 0.0],
                                            [0.019375287645268975, 0.020202367947773765, 0.0],
                                            [0.0, 0.0, 0.014056764993693264],
                                        ],
                                        [
                                            [0.0, 0.0, 0.036007006646053566],
                                            [0.0, 0.0, 0.02824399694443206],
                                            [0.03600700664605357, 0.028243996944432068, 0.0],
                                        ],
                                    ],
                                    [
                                        [
                                            [-0.05593597453787985, 0.01937528764526898, 0.0],
                                            [0.019375287645269, 0.020202367947773803, 0.0],
                                            [0.0, 0.0, 0.014056764993693242],
                                        ],
                                        [
                                            [0.30428426262352537, 0.020202367947773744, 0.0],
                                            [0.020202367947773803, 0.4816266918021246, 0.0],
                                            [0.0, 0.0, 0.40269913477850905],
                                        ],
                                        [
                                            [0.0, 0.0, 0.02824399694443207],
                                            [0.0, 0.0, 0.05649509961573344],
                                            [0.02824399694443207, 0.05649509961573344, 0.0],
                                        ],
                                    ],
                                    [
                                        [
                                            [0.0, 0.0, 0.03600700664605357],
                                            [0.0, 0.0, 0.028243996944432068],
                                            [0.03600700664605358, 0.028243996944432075, 0.0],
                                        ],
                                        [
                                            [0.0, 0.0, 0.028243996944432078],
                                            [0.0, 0.0, 0.05649509961573346],
                                            [0.028243996944432078, 0.05649509961573345, 0.0],
                                        ],
                                        [
                                            [0.36650820953365676, 0.01405676499369331, 0.0],
                                            [0.014056764993693339, 0.402699134778509, 0.0],
                                            [0.0, 0.0, 0.4498590410866706],
                                        ],
                                    ],
                                ]
                            ),
                            None,
                            None,
                            None,
                        ),
                    ),
                ],
                energy_shift={"ActiveSpaceTransformer": -14.253802923103054},
            )
        )

        self.assertDriverResult(driver_result_reduced, expected)

    def test_arbitrary_active_orbitals(self):
        """Test manual selection of active orbital indices."""
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        trafo = ActiveSpaceTransformer(
            num_electrons=2, num_molecular_orbitals=2, active_orbitals=[0, 2]
        )
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicStructureProblem(
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
        expected.properties.electronic_dipole_moment = ElectronicDipoleMoment(
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
        driver = PySCFDriver()
        driver_result = driver.run()

        with self.assertRaises(QiskitNatureError, msg=message):
            ActiveSpaceTransformer(
                num_electrons=num_electrons,
                num_molecular_orbitals=num_molecular_orbitals,
                active_orbitals=active_orbitals,
            ).transform(driver_result)

    def test_tuple_num_electrons_with_manual_orbitals(self):
        """Regression test against https://github.com/Qiskit/qiskit-nature/issues/434."""
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        trafo = ActiveSpaceTransformer(
            num_electrons=(1, 1),
            num_molecular_orbitals=2,
            active_orbitals=[0, 1],
        )
        driver_result_reduced = trafo.transform(driver_result)

        expected = ElectronicStructureProblem(
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
        expected.properties.electronic_dipole_moment = ElectronicDipoleMoment(
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
                                np.asarray([[0.69447435, -1.01418298], [-1.01418298, 0.69447435]]),
                                None,
                            ),
                        )
                    ],
                    shift={"ActiveSpaceTransformer": 0.0},
                ),
            ]
        )

        self.assertDriverResult(driver_result_reduced, expected)

    def test_no_deep_copy(self):
        """Test that objects are not being deeply copied.

        This is a regression test against the fix applied by
        https://github.com/Qiskit/qiskit-nature/pull/659
        """
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)
        driver_result_reduced = trafo.transform(driver_result)

        active_transform = np.asarray(
            [
                [0.32774803333032304, 0.12166492852424596],
                [0.27055282555225113, 1.7276386116201712],
                [0.32774803333032265, -0.12166492852424832],
                [0.2705528255522547, -1.727638611620168],
            ]
        )

        np.testing.assert_array_almost_equal(
            np.abs(driver_result_reduced.basis_transform.coeff_alpha),
            np.abs(active_transform),
        )

    def test_numpy_integer(self):
        """Tests that numpy integer objects do not cause issues in `isinstance` checks.

        This is a regression test against the fix applied by
        https://github.com/Qiskit/qiskit-nature/pull/712
        """
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        particle_number = driver_result.properties.particle_number
        driver_result.properties.particle_number = None
        particle_number.num_alpha = np.int64(particle_number.num_alpha)
        particle_number.num_beta = np.int64(particle_number.num_beta)
        particle_number.num_spin_orbitals = np.int64(particle_number.num_spin_orbitals)

        driver_result.properties.particle_number = particle_number

        trafo = ActiveSpaceTransformer(
            num_electrons=particle_number.num_particles, num_molecular_orbitals=2
        )
        _ = trafo.transform(driver_result)


if __name__ == "__main__":
    unittest.main()
