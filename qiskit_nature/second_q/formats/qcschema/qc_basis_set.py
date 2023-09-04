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

"""The QCSchema basis set dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import h5py

from .qc_base import _QCBase


@dataclass
class QCElectronShell(_QCBase):
    """A dataclass to store the information of a single electron shell in a basis set.

    For more information refer to
    [here](https://github.com/MolSSI/QCSchema/blob/1d5ff3baa5/qcschema/dev/definitions.py#L43).
    """

    angular_momentum: Sequence[int]
    """The angular momenta of this electron shell as a list of integers."""
    harmonic_type: str
    """The type of this shell."""
    exponents: Sequence[float | str]
    """The exponents of this contracted shell. The official spec stores these values as strings."""
    coefficients: Sequence[Sequence[float | str]]
    """The general contraction coefficients of this contracted shell. The official spec stores these
    values as strings."""

    def to_hdf5(self, group: h5py.Group) -> None:
        group.attrs["angular_momentum"] = self.angular_momentum
        group.attrs["harmonic_type"] = self.harmonic_type
        group.create_dataset("exponents", data=self.exponents)
        group.create_dataset("coefficients", data=self.coefficients)


@dataclass
class QCECPPotential(_QCBase):
    """A dataclass to store the information of an ECP in a basis set.

    For more information refer to
    [here](https://github.com/MolSSI/QCSchema/blob/1d5ff3baa5/qcschema/dev/definitions.py#L90).
    """

    ecp_type: str
    """The type of this potential."""
    angular_momentum: Sequence[int]
    """The angular momenta of this potential as a list of integers."""
    r_exponents: Sequence[int]
    """The exponents of the `r` term."""
    gaussian_exponents: Sequence[float | str]
    """The exponents of the gaussian terms. The official spec stores these values as strings."""
    coefficients: Sequence[Sequence[float | str]]
    """The general contraction coefficients of this potential. The official spec stores these values
    as strings."""

    def to_hdf5(self, group: h5py.Group) -> None:
        group.attrs["angular_momentum"] = self.angular_momentum
        group.attrs["ecp_type"] = self.ecp_type
        group.create_dataset("r_exponents", data=self.r_exponents)
        group.create_dataset("gaussian_exponents", data=self.gaussian_exponents)
        group.create_dataset("coefficients", data=self.coefficients)


@dataclass
class QCCenterData(_QCBase):
    """A dataclass to store the information of a single atom/center in the basis set.

    For more information refer to
    [here](https://github.com/MolSSI/QCSchema/blob/1d5ff3baa5/qcschema/dev/definitions.py#L146).
    """

    electron_shells: Sequence[QCElectronShell] | None = None
    """The list of electronic shells for this element."""
    ecp_electrons: int | None = None
    """The number of electrons replaced by an ECP."""
    ecp_potentials: Sequence[QCECPPotential] | None = None
    """The list of effective core potentials for this element."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCCenterData:
        electron_shells: Sequence[QCElectronShell] | None = None
        if "electron_shells" in data.keys():
            electron_shells = []
            for shell in data.pop("electron_shells", []):
                electron_shells.append(cast(QCElectronShell, QCElectronShell.from_dict(shell)))

        ecp_potentials: Sequence[QCECPPotential] | None = None
        if "ecp_potentials" in data.keys():
            ecp_potentials = []
            for ecp in data.pop("ecp_potentials", []):
                ecp_potentials.append(cast(QCECPPotential, QCECPPotential.from_dict(ecp)))

        return cls(**data, electron_shells=electron_shells, ecp_potentials=ecp_potentials)

    def to_hdf5(self, group: h5py.Group) -> None:
        if self.electron_shells:
            electron_shells = group.require_group("electron_shells")
            for idx, shell in enumerate(self.electron_shells):
                idx_group = electron_shells.create_group(str(idx))
                shell.to_hdf5(idx_group)

        if self.ecp_electrons is not None:
            group.attrs["ecp_electrons"] = self.ecp_electrons

        if self.ecp_potentials:
            ecp_potentials = group.require_group("ecp_potentials")
            for idx, ecp in enumerate(self.ecp_potentials):
                idx_group = ecp_potentials.create_group(str(idx))
                ecp.to_hdf5(idx_group)

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCCenterData:
        electron_shells: Sequence[QCElectronShell] | None = None
        if "electron_shells" in h5py_group.keys():
            electron_shells = []
            for shell in h5py_group["electron_shells"].values():
                electron_shells.append(cast(QCElectronShell, QCElectronShell.from_hdf5(shell)))

        ecp_potentials: Sequence[QCECPPotential] | None = None
        if "ecp_potentials" in h5py_group.keys():
            ecp_potentials = []
            for ecp in h5py_group["ecp_potentials"].values():
                ecp_potentials.append(cast(QCECPPotential, QCECPPotential.from_hdf5(ecp)))

        return cls(
            electron_shells=electron_shells,
            ecp_electrons=h5py_group.attrs.get("ecp_electrons", None),
            ecp_potentials=ecp_potentials,
        )


@dataclass
class QCBasisSet(_QCBase):
    """A dataclass to store the information of the basis set used in the original calculation.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_basis.html#basis-set-schema).
    """

    center_data: Mapping[str, QCCenterData]
    """A dictionary mapping the keys provided by `atom_map` to their basis center data."""
    atom_map: Sequence[str]
    """The list of atomic kinds, indicating the keys used to store the basis in `center_data`."""
    name: str
    """The name of the basis set."""
    schema_version: int | None = None
    """The version of this specific schema."""
    schema_name: str | None = None
    """The name of this schema. This value is expected to be `qcschema_basis`."""
    description: str | None = None
    """A description of this basis set."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCBasisSet:
        center_data = {k: QCCenterData.from_dict(v) for k, v in data.pop("center_data").items()}
        return cls(**data, center_data=center_data)

    def to_hdf5(self, group: h5py.Group) -> None:
        center_data = group.require_group("center_data")
        for key, value in self.center_data.items():
            key_group = center_data.require_group(key)
            value.to_hdf5(key_group)

        group.attrs["atom_map"] = self.atom_map
        group.attrs["name"] = self.name
        if self.schema_version is not None:
            group.attrs["schema_version"] = self.schema_version
        if self.schema_name is not None:
            group.attrs["schema_name"] = self.schema_name
        if self.description is not None:
            group.attrs["description"] = self.description

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCBasisSet:
        center_data: dict[str, QCCenterData] = {}
        for name, group in h5py_group["center_data"].items():
            center_data[name] = cast(QCCenterData, QCCenterData.from_hdf5(group))

        return cls(
            center_data=center_data,
            atom_map=h5py_group.attrs["atom_map"],
            name=h5py_group.attrs["name"],
            schema_version=h5py_group.attrs.get("schema_version", None),
            schema_name=h5py_group.attrs.get("schema_name", None),
            description=h5py_group.attrs.get("description", None),
        )
