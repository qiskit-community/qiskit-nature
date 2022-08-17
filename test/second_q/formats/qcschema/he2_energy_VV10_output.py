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

"""The expected He2 energy QCSchema."""
# pylint: disable=invalid-name

from qiskit_nature.second_q.formats.qcschema import (
    QCModel,
    QCProperties,
    QCProvenance,
    QCSchema,
    QCTopology,
)

EXPECTED = QCSchema(
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
