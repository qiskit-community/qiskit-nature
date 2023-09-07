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

"""The QCSchema (output) dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

# Sphinx is somehow unable to resolve this forward reference which gets inherited from QCSchemaInput
# Thus, we import it here manually to ensure the documentation can be built
from typing import Mapping  # pylint: disable=unused-import

import h5py

from qiskit_nature.version import __version__

from .qc_error import QCError
from .qc_model import QCModel
from .qc_schema_input import QCSchemaInput
from .qc_properties import QCProperties
from .qc_provenance import QCProvenance
from .qc_topology import QCTopology
from .qc_wavefunction import QCWavefunction


@dataclass
class QCSchema(QCSchemaInput):
    """The full QCSchema as a dataclass.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#output-components).
    """

    provenance: QCProvenance
    """An instance of :class:`QCProvenance`."""
    return_result: float | Sequence[float]
    """The primary result of the computation. Its value depends on the type of computation (see also
    `driver`)."""
    success: bool
    """Whether the computation was successful."""
    properties: QCProperties
    """An instance of :class:`QCProperties`."""
    error: QCError | None = None
    """An instance of :class:`QCError` if the computation was not successful (`success = False`)."""
    wavefunction: QCWavefunction | None = None
    """An instance of :class:`QCWavefunction`."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCSchema:
        error: QCError | None = None
        if "error" in data.keys():
            error = QCError(**data.pop("error"))
        model = QCModel(**data.pop("model"))
        molecule = QCTopology(**data.pop("molecule"))
        provenance = QCProvenance(**data.pop("provenance"))
        properties = QCProperties(**data.pop("properties"))
        wavefunction: QCWavefunction | None = None
        if "wavefunction" in data.keys():
            wavefunction = QCWavefunction.from_dict(data.pop("wavefunction"))
        return cls(
            **data,
            error=error,
            model=model,
            molecule=molecule,
            provenance=provenance,
            properties=properties,
            wavefunction=wavefunction,
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        group.attrs["schema_name"] = self.schema_name
        group.attrs["schema_version"] = self.schema_version
        group.attrs["driver"] = self.driver
        group.attrs["return_result"] = self.return_result
        group.attrs["success"] = self.success

        molecule_group = group.require_group("molecule")
        self.molecule.to_hdf5(molecule_group)

        model_group = group.require_group("model")
        self.model.to_hdf5(model_group)

        provenance_group = group.require_group("provenance")
        self.provenance.to_hdf5(provenance_group)

        properties_group = group.require_group("properties")
        self.properties.to_hdf5(properties_group)

        if self.error is not None:
            error_group = group.require_group("error")
            self.error.to_hdf5(error_group)

        if self.wavefunction is not None:
            wavefunction_group = group.require_group("wavefunction")
            self.wavefunction.to_hdf5(wavefunction_group)

        keywords_group = group.require_group("keywords")
        for key, value in self.keywords.items():
            keywords_group.attrs[key] = value

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCSchemaInput:
        data = dict(h5py_group.attrs.items())

        data["molecule"] = cast(QCTopology, QCTopology.from_hdf5(h5py_group["molecule"]))
        data["model"] = cast(QCModel, QCModel.from_hdf5(h5py_group["model"]))
        data["provenance"] = cast(QCProvenance, QCProvenance.from_hdf5(h5py_group["provenance"]))
        data["properties"] = cast(QCProperties, QCProperties.from_hdf5(h5py_group["properties"]))

        if "error" in h5py_group.keys():
            data["error"] = cast(QCError, QCError.from_hdf5(h5py_group["error"]))

        if "wavefunction" in h5py_group.keys():
            data["wavefunction"] = cast(
                QCWavefunction, QCWavefunction.from_hdf5(h5py_group["wavefunction"])
            )

        data["keywords"] = dict(h5py_group["keywords"].attrs.items())

        return cls(**data)
