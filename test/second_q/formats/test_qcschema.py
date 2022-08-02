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

"""Test the QCSchema implementation."""

import unittest

from test import QiskitNatureTestCase

from qiskit_nature.second_q.formats.qcschema import (
    QCBasisSet,
    QCCenterData,
    QCElectronShell,
    QCModel,
    QCProperties,
    QCProvenance,
    QCSchemaInput,
    QCSchema,
    QCTopology,
    QCWavefunction,
)


class TestRealHe2Data(QiskitNatureTestCase):
    """Tests the QCSchema on real He2 data obtained from the QCSchema repository."""

    FILENAME = "he2_energy_VV10"

    EXPECTED_INPUT = QCSchemaInput(
        schema_name="qc_schema_input",
        schema_version=1,
        molecule=QCTopology(
            symbols=["He", "He"],
            geometry=[0, 0, 0, 0, 0, 6],
            schema_name="qcschema_molecule",
            schema_version=2,
        ),
        driver="energy",
        model=QCModel(
            method="VV10",
            basis="cc-pVDZ",
        ),
        keywords={},
    )

    EXPECTED_OUTPUT = QCSchema(
        schema_name="qc_schema_output",
        schema_version=1,
        molecule=QCTopology(
            symbols=["He", "He"],
            geometry=[0, 0, 0, 0, 0, 6],
            schema_name="qcschema_molecule",
            schema_version=2,
        ),
        driver="energy",
        model=QCModel(
            method="VV10",
            basis="cc-pVDZ",
        ),
        keywords={},
        provenance=QCProvenance(
            creator="QM Program",
            version="1.1",
            routine="module.json.run_json",
        ),
        return_result=-5.815121364568496,
        success=True,
        properties=QCProperties(
            calcinfo_nbasis=10,
            calcinfo_nmo=10,
            calcinfo_nalpha=2,
            calcinfo_nbeta=2,
            calcinfo_natom=2,
            return_energy=-5.815121364568496,
            scf_one_electron_energy=-9.10156722786234,
            scf_two_electron_energy=4.782528510470115,
            nuclear_repulsion_energy=0.6666666666666666,
            scf_vv10_energy=0.018799951240226136,
            scf_xc_energy=-2.181549265083163,
            scf_dipole_moment=[0.0, 0.0, 9.030096599360606e-14],  # type: ignore[arg-type]
            scf_total_energy=-5.815121364568496,
            scf_iterations=3,
        ),
    )

    def test_input(self):
        """Tests the input parsing."""
        file = self.get_resource_path(self.FILENAME + "_input.json", "second_q/formats/qcschema")
        qcs = QCSchemaInput.from_json(file)
        self.assertEqual(qcs, self.EXPECTED_INPUT)

    def test_output(self):
        """Tests the output parsing."""
        file = self.get_resource_path(self.FILENAME + "_output.json", "second_q/formats/qcschema")
        qcs = QCSchema.from_json(file)
        self.assertEqual(qcs, self.EXPECTED_OUTPUT)


class TestRealWaterData(QiskitNatureTestCase):
    """Tests the QCSchema on real Water data obtained from the QCSchema repository."""

    EXPECTED = QCSchema(
        schema_name="qc_schema_output",
        schema_version=1,
        molecule=QCTopology(
            symbols=["O", "H", "H"],
            geometry=[
                0.0,
                0.0,
                -0.1294769411935893,
                0.0,
                -1.494187339479985,
                1.0274465079245698,
                0.0,
                1.494187339479985,
                1.0274465079245698,
            ],
            schema_name="qcschema_molecule",
            schema_version=2,
        ),
        driver="energy",
        model=QCModel(
            method="B3LYP",
            basis="cc-pVDZ",
        ),
        keywords={},
        provenance=QCProvenance(
            creator="QM Program",
            version="1.1",
            routine="module.json.run_json",
        ),
        return_result=-76.4187620271478,
        success=True,
        properties=QCProperties(
            calcinfo_nbasis=24,
            calcinfo_nmo=24,
            calcinfo_nalpha=5,
            calcinfo_nbeta=5,
            calcinfo_natom=3,
            return_energy=-76.4187620271478,
            scf_one_electron_energy=-122.5182981454265,
            scf_two_electron_energy=44.844942513688004,
            nuclear_repulsion_energy=8.80146205625184,
            scf_dipole_moment=[0.0, 0.0, 1.925357619589245],  # type: ignore[arg-type]
            scf_total_energy=-76.4187620271478,
            scf_xc_energy=-7.546868451661161,
            scf_iterations=6,
        ),
        wavefunction=QCWavefunction(
            basis=QCBasisSet(
                name="6-31G",
                description="6-31G on all Hydrogen and Oxygen atoms",
                center_data={
                    "bs_631g_h": QCCenterData(
                        electron_shells=[
                            QCElectronShell(
                                harmonic_type="spherical",
                                angular_momentum=[0],
                                exponents=["18.731137", "2.8253944", "0.6401217"],
                                coefficients=[["0.0334946", "0.2347269", "0.8137573"]],
                            ),
                            QCElectronShell(
                                harmonic_type="spherical",
                                angular_momentum=[0],
                                exponents=["0.1612778"],
                                coefficients=[["1.0000000"]],
                            ),
                        ],
                    ),
                    "bs_631g_o": QCCenterData(
                        electron_shells=[
                            QCElectronShell(
                                harmonic_type="spherical",
                                angular_momentum=[0],
                                exponents=[
                                    "5484.6717000",
                                    "825.2349500",
                                    "188.0469600",
                                    "52.9645000",
                                    "16.8975700",
                                    "5.7996353",
                                ],
                                coefficients=[
                                    [
                                        "0.0018311",
                                        "0.0139501",
                                        "0.0684451",
                                        "0.2327143",
                                        "0.4701930",
                                        "0.3585209",
                                    ]
                                ],
                            ),
                            QCElectronShell(
                                harmonic_type="spherical",
                                angular_momentum=[0, 1],
                                exponents=["15.5396160", "3.5999336", "1.0137618"],
                                coefficients=[
                                    ["-0.1107775", "-0.1480263", "1.1307670"],
                                    ["0.0708743", "0.3397528", "0.7271586"],
                                ],
                            ),
                            QCElectronShell(
                                harmonic_type="spherical",
                                angular_momentum=[0, 1],
                                exponents=["0.2700058"],
                                coefficients=[["1.0000000"], ["1.0000000"]],
                            ),
                        ],
                    ),
                },
                atom_map=[
                    "bs_631g_o",
                    "bs_631g_h",
                    "bs_631g_h",
                ],
            ),
        ),
    )

    def test_water_output_v1(self):
        """Tests the v1 water output"""
        file = self.get_resource_path("water_output.json", "second_q/formats/qcschema")
        qcs = QCSchema.from_json(file)
        self.assertEqual(qcs, self.EXPECTED)

    def test_water_output_v3(self):
        """Tests the v3 water output"""
        file = self.get_resource_path("water_output_v3.json", "second_q/formats/qcschema")
        qcs = QCSchema.from_json(file)

        expected_v3 = self.EXPECTED
        expected_v3.schema_version = 3
        expected_v3.wavefunction.restricted = True

        self.assertEqual(qcs, expected_v3)


if __name__ == "__main__":
    unittest.main()
