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

"""The expected He2 energy QCSchemaInput."""
# pylint: disable=invalid-name

from qiskit_nature.second_q.formats.qcschema import (
    QCModel,
    QCSchemaInput,
    QCTopology,
)

EXPECTED = QCSchemaInput(
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
