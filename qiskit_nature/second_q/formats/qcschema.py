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

"""The QCSchema format.

The documentation of this schema can be found
[here](https://molssi-qc-schema.readthedocs.io/en/latest/).
"""
# pylint: disable=invalid-name

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import json
import h5py


class _QCBase:
    """A base class for the QCSchema dataclasses.

    This base class is used to implement schema-wide conversion utility methods.
    """

    def to_dict(self) -> dict[str, Any]:
        """Converts the schema object to a dictionary.

        Returns:
            The dictionary representation of the schema object.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Converts the schema object to JSON.

        Returns:
            The JSON representation of the schema object.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _QCBase:
        """Constructs a schema object from a dictionary of data.

        The dictionary provided to this method corresponds to the format as obtained by `json.load`
        from a JSON representation of the schema object according to the latest standard as
        documented [here](https://molssi-qc-schema.readthedocs.io/en/latest/).

        Args:
            data: the data dictionary.

        Returns:
            An instance of the schema object.
        """
        return cls(**data)

    @classmethod
    def from_json(cls, fp: str | Path) -> _QCBase:
        """Constructs a schema object from a JSON file.

        The JSON data stored in the file must match the latest standard as documented
        [here](https://molssi-qc-schema.readthedocs.io/en/latest/).

        Args:
            fp: the path to the JSON file.

        Returns:
            An instance of the schema object.
        """
        with open(fp, "r", encoding="utf8") as file:
            return cls.from_dict(json.load(file))

    def to_hdf5(self, group: h5py.Group) -> None:
        """Converts the schema object to HDF5.

        Args:
            group: the h5py group into which to store the object.
        """
        # we use __dict__ here because we do not want the recursive behavior of asdict()
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if hasattr(value, "to_hdf5"):
                inner_group = group.require_group(key)
                value.to_hdf5(inner_group)
            else:
                group.attrs[key] = value

    @classmethod
    def from_hdf5(cls, h5py_data: str | Path | h5py.Group) -> _QCBase:
        """Constructs a schema object from an HDF5 object.

        While the QCSchema is officially tailored to support JSON, HDF5 is supported as a more
        high-performance alternative and considered the standard within Qiskit Nature. Due to its
        similarities with JSON a 1-to-1 correspondence can be made between the two.

        For more details refer to
        [here](https://molssi-qc-schema.readthedocs.io/en/latest/tech_discussion.html#json-and-hdf5).

        Args:
            h5py_data: can be either the path to a file or an `h5py.Group`.

        Returns:
            An instance of the schema object.
        """
        if isinstance(h5py_data, h5py.Group):
            return cls._from_hdf5_group(h5py_data)

        with h5py.File(h5py_data, "r") as file:
            return cls._from_hdf5_group(file)

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> _QCBase:
        """This internal method deals with actually constructing a schema object from an `h5py.Group`.

        Args:
            h5py_group: the actual `h5py.Group`.

        Returns:
            An instance of the schema object.
        """
        data = dict(h5py_group.attrs.items())
        for key, value in h5py_group.items():
            data[key] = value[...]
        return cls(**data)


@dataclass
class QCError(_QCBase):
    """A dataclass to store the failure information contained in a QCSchema.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#success).
    """

    error_type: str
    error_message: str


@dataclass
class QCProvenance(_QCBase):
    """A dataclass to store the program information that generated the QCSchema file.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#provenance).
    """

    creator: str
    version: str
    routine: str


@dataclass
class QCModel(_QCBase):
    """A dataclass to store the mathematical model information used in the original calculation.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#model).
    """

    method: str
    basis: str | QCBasisSet

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCModel:
        basis: str | dict[str, Any] | QCBasisSet = data.pop("basis")
        if isinstance(basis, dict):
            basis = QCBasisSet.from_dict(basis)
        return cls(**data, basis=basis)

    def to_hdf5(self, group: h5py.Group) -> None:
        if isinstance(self.basis, QCBasisSet):
            basis_group = group.require_group("basis")
            self.basis.to_hdf5(basis_group)
        else:
            group.attrs["basis"] = self.basis

        group.attrs["method"] = self.method

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCModel:
        basis: str | QCBasisSet
        if "basis" in h5py_group.keys():
            basis = cast(QCBasisSet, QCBasisSet.from_hdf5(h5py_group["basis"]))
        else:
            basis = h5py_group.attrs["basis"]

        return cls(
            method=h5py_group.attrs["method"],
            basis=basis,
        )


@dataclass
class QCTopology(_QCBase):
    """A dataclass to store the topological information of the physical system.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#topology).
    """

    symbols: list[str]
    geometry: list[float]
    schema_name: str
    schema_version: int
    molecular_charge: int = None
    molecular_multiplicity: int = None
    fix_com: bool = None
    real: list[bool] = None
    connectivity: list[tuple[int, int, int]] = None
    fix_orientation: bool = None
    atom_labels: list[str] = None
    fragment_multiplicities: list[int] = None
    fix_symmetry: str = None
    fragment_charges: list[float] = None
    mass_numbers: list[int] = None
    name: str = None
    masses: list[float] = None
    comment: str = None
    provenance: QCProvenance = None
    fragments: list[tuple[int, ...]] = None
    atomic_numbers: list[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCTopology:
        provenance: QCProvenance = None
        if "provenance" in data.keys():
            provenance = cast(QCProvenance, QCProvenance.from_dict(data.pop("provenance")))
        return cls(**data, provenance=provenance)

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCTopology:
        data = dict(h5py_group.attrs.items())

        for key, value in h5py_group.items():
            data[key] = value

        if "provenance" in h5py_group.keys():
            data["provenance"] = cast(
                QCProvenance, QCProvenance.from_hdf5(h5py_group["provenance"])
            )

        return cls(**data)


@dataclass
class QCElectronShell(_QCBase):
    """A dataclass to store the information of a single electron shell in a basis set.

    For more information refer to
    [here](https://github.com/MolSSI/QCSchema/blob/1d5ff3baa5/qcschema/dev/definitions.py#L43).
    """

    angular_momentum: list[int]
    harmonic_type: str
    exponents: list[float | str]
    coefficients: list[list[float | str]]

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
    angular_momentum: list[int]
    r_exponents: list[int]
    gaussian_exponents: list[float | str]
    coefficients: list[list[float | str]]

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

    electron_shells: list[QCElectronShell] = None
    ecp_electrons: int = None
    ecp_potentials: list[QCECPPotential] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCCenterData:
        electron_shells: list[QCElectronShell] = None
        if "electron_shells" in data.keys():
            electron_shells = []
            for shell in data.pop("electron_shells", []):
                electron_shells.append(cast(QCElectronShell, QCElectronShell.from_dict(shell)))

        ecp_potentials: list[QCECPPotential] = None
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
        electron_shells: list[QCElectronShell] = None
        if "electron_shells" in h5py_group.keys():
            electron_shells = []
            for shell in h5py_group["electron_shells"].values():
                electron_shells.append(cast(QCElectronShell, QCElectronShell.from_hdf5(shell)))

        ecp_potentials: list[QCECPPotential] = None
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

    center_data: dict[str, QCCenterData]
    atom_map: list[str]
    name: str
    schema_version: int = None
    schema_name: str = None
    description: str = None

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


@dataclass
class QCProperties(_QCBase):
    """A dataclass to store the computed properties of the original calculation.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_props.html#properties-schema).
    """

    calcinfo_nbasis: int = None
    calcinfo_nmo: int = None
    calcinfo_nalpha: int = None
    calcinfo_nbeta: int = None
    calcinfo_natom: int = None
    return_energy: float = None

    scf_one_electron_energy: float = None
    scf_two_electron_energy: float = None
    nuclear_repulsion_energy: float = None
    scf_vv10_energy: float = None
    scf_xc_energy: float = None
    scf_dispersion_correction_energy: float = None
    scf_dipole_moment: tuple[float, float, float] = None
    scf_total_energy: float = None
    scf_iterations: int = None

    mp2_same_spin_correlation_energy: float = None
    mp2_opposite_spin_correlation_energy: float = None
    mp2_singles_energy: float = None
    mp2_doubles_energy: float = None
    mp2_correlation_energy: float = None
    mp2_total_energy: float = None
    mp2_dipole_moment: tuple[float, float, float] = None

    ccsd_same_spin_correlation_energy: float = None
    ccsd_opposite_spin_correlation_energy: float = None
    ccsd_singles_energy: float = None
    ccsd_doubles_energy: float = None
    ccsd_correlation_energy: float = None
    ccsd_total_energy: float = None
    ccsd_prt_pr_correlation_energy: float = None
    ccsd_prt_pr_total_energy: float = None
    ccsdt_correlation_energy: float = None
    ccsdt_total_energy: float = None
    ccsdtq_correlation_energy: float = None
    ccsdtq_total_energy: float = None
    ccsd_dipole_moment: tuple[float, float, float] = None
    ccsd_prt_pr_dipole_moment: tuple[float, float, float] = None
    ccsdt_dipole_moment: tuple[float, float, float] = None
    ccsdtq_dipole_moment: tuple[float, float, float] = None
    ccsd_iterations: int = None
    ccsdt_iterations: int = None
    ccsdtq_iterations: int = None


@dataclass
class QCWavefunction(_QCBase):
    """A dataclass to store any additional computed wavefunction properties.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_wf.html#wavefunction-schema).
    """

    basis: QCBasisSet

    orbitals_a: str = None
    orbitals_b: str = None
    density_a: str = None
    density_b: str = None
    fock_a: str = None
    fock_b: str = None
    eigenvalues_a: str = None
    eigenvalues_b: str = None
    occupations_a: str = None
    occupations_b: str = None

    scf_orbitals_a: list[float] = None
    scf_orbitals_b: list[float] = None
    scf_density_a: list[float] = None
    scf_density_b: list[float] = None
    scf_fock_a: list[float] = None
    scf_fock_b: list[float] = None
    scf_coulomb_a: list[float] = None
    scf_coulomb_b: list[float] = None
    scf_exchange_a: list[float] = None
    scf_exchange_b: list[float] = None
    scf_eigenvalues_a: list[float] = None
    scf_eigenvalues_b: list[float] = None
    scf_occupations_a: list[float] = None
    scf_occupations_b: list[float] = None

    localized_orbitals_a: list[float] = None
    localized_orbitals_b: list[float] = None
    localized_fock_a: list[float] = None
    localized_fock_b: list[float] = None

    h_core_a: list[float] = None
    h_core_b: list[float] = None
    h_effective_a: list[float] = None
    h_effective_b: list[float] = None

    restricted: bool = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCWavefunction:
        basis = QCBasisSet.from_dict(data.pop("basis"))
        return cls(**data, basis=basis)

    def to_hdf5(self, group: h5py.Group) -> None:
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key == "restricted":
                group.attrs["restricted"] = self.restricted
            elif hasattr(value, "to_hdf5"):
                inner_group = group.require_group(key)
                value.to_hdf5(inner_group)
            else:
                group.create_dataset(key, data=value)

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> QCWavefunction:
        data = dict(h5py_group.attrs.items())

        for key, value in h5py_group.items():
            if key == "basis":
                data["basis"] = cast(QCBasisSet, QCBasisSet.from_hdf5(h5py_group["basis"]))
            else:
                data[key] = value[...]

        return cls(**data)


@dataclass
class QCSchemaInput(_QCBase):
    """A dataclass containing all *classical input* components of the QCSchema.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#input-components).
    """

    schema_name: str
    schema_version: int
    molecule: QCTopology
    driver: str
    model: QCModel
    keywords: dict[str, Any]

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


@dataclass
class QCSchema(QCSchemaInput):
    """The full QCSchema as a dataclass.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#output-components).
    """

    provenance: QCProvenance
    return_result: float | list[float]
    success: bool
    properties: QCProperties
    error: QCError = None
    wavefunction: QCWavefunction = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCSchema:
        error: QCError = None
        if "error" in data.keys():
            error = QCError(**data.pop("error"))
        model = QCModel(**data.pop("model"))
        molecule = QCTopology(**data.pop("molecule"))
        provenance = QCProvenance(**data.pop("provenance"))
        properties = QCProperties(**data.pop("properties"))
        wavefunction: QCWavefunction = None
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
