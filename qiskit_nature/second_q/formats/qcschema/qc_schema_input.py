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

"""The QCSchema input dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import h5py

from .qc_base import _QCBase
from .qc_model import QCModel
from .qc_topology import QCTopology


@dataclass
class QCSchemaInput(_QCBase):
    """A dataclass containing all *classical input* components of the QCSchema.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#input-components).
    """

    schema_name: str
    """The name of this schema. This value is expected to be `qcschema` or `qc_schema`."""
    schema_version: int
    """The version of this specific schema."""
    molecule: QCTopology
    """An instance of :class:`QCTopology`."""
    driver: str
    """The type of computation. Example values are `energy`, `gradient`, and `hessian`."""
    model: QCModel
    """An instance of :class:`QCModel`."""
    keywords: Mapping[str, Any]
    """Any additional program-specific parameters."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCSchemaInput:
        model = QCModel(**data.pop("model"))
        molecule = QCTopology(**data.pop("molecule"))
        return cls(**data, model=model, molecule=molecule)

    def to_hdf5(self, group: h5py.Group) -> None:
        group.attrs["schema_name"] = self.schema_name
        group.attrs["schema_version"] = self.schema_version
        group.attrs["driver"] = self.driver

        molecule_group = group.require_group("molecule")
        self.molecule.to_hdf5(molecule_group)

        model_group = group.require_group("model")
        self.model.to_hdf5(model_group)

        keywords_group = group.require_group("keywords")
        for key, value in self.keywords.items():
            keywords_group.attrs[key] = value

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCSchemaInput:
        data = dict(h5py_group.attrs.items())

        data["molecule"] = cast(QCTopology, QCTopology.from_hdf5(h5py_group["molecule"]))
        data["model"] = cast(QCModel, QCModel.from_hdf5(h5py_group["model"]))
        data["keywords"] = dict(h5py_group["keywords"].attrs.items())

        return cls(**data)
