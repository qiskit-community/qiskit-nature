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

"""The expected water output QCSchema."""

from qiskit_nature.second_q.formats.qcschema import (
    QCBasisSet,
    QCCenterData,
    QCElectronShell,
    QCModel,
    QCProperties,
    QCProvenance,
    QCSchema,
    QCTopology,
    QCWavefunction,
)

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
