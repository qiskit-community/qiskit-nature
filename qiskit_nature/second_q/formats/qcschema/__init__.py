# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
QCSchema (:mod:`qiskit_nature.second_q.formats.qcschema`)
=============================================================

The documentation of this schema can be found
[here](https://molssi-qc-schema.readthedocs.io/en/latest/).

In our Python implementation of this schema, we are handling optional attributes of the schema a bit
differently than how they work in a JSON or HDF5 container. Whereas in these latter cases, the
attributes might not simply exist at all, in our Python data classes, the attributes will always be
available but might simply take the value `None`.
When dumping a Python instance of this schema to either file format, attributes which are `None`
will be filtered, in order to ensure that loading and dumping the same data subsequently, does not
change the available attributes.

It should also be noted, that optional attributes will not take "default" values on the schema
object. Thus, it is up to the code reading from a schema instance, to handle `None` i.e. undefined
values for a specific variable, for example using in its place a suitable default value instead.

.. currentmodule:: qiskit_nature.second_q.formats.qcschema

"""


from .qc_basis_set import QCBasisSet, QCCenterData, QCECPPotential, QCElectronShell
from .qc_error import QCError
from .qc_model import QCModel
from .qc_properties import QCProperties
from .qc_provenance import QCProvenance
from .qc_schema import QCSchema
from .qc_schema_input import QCSchemaInput
from .qc_topology import QCTopology
from .qc_wavefunction import QCWavefunction

__all__ = [
    "QCBasisSet",
    "QCCenterData",
    "QCECPPotential",
    "QCElectronShell",
    "QCError",
    "QCModel",
    "QCProperties",
    "QCProvenance",
    "QCSchema",
    "QCSchemaInput",
    "QCTopology",
    "QCWavefunction",
]
