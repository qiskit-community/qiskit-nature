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

"""
=============================================================
The QCSchema (:mod:`qiskit_nature.second_q.formats.qcschema`)
=============================================================

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

        def filter_none(d: list[tuple[str, Any]]) -> dict[str, Any]:
            return {k: v for (k, v) in d if v is not None}

        return asdict(self, dict_factory=filter_none)

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

    def to_json(self) -> str:
        """Converts the schema object to JSON.

        Returns:
            The JSON representation of the schema object.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_data: str | bytes | Path) -> _QCBase:
        """Constructs a schema object from JSON.

        The JSON data must match the latest standard as documented
        [here](https://molssi-qc-schema.readthedocs.io/en/latest/).

        Args:
            json_data: can be either the path to a file or the json data directly provided as a `str`.

        Returns:
            An instance of the schema object.
        """
        try:
            return cls.from_dict(json.loads(json_data))  # type: ignore[arg-type]
        except json.JSONDecodeError:
            with open(json_data, "r", encoding="utf8") as file:
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
    """The type of error that was raised."""
    error_message: str
    """A description of the error that was raised."""


@dataclass
class QCProvenance(_QCBase):
    """A dataclass to store the program information that generated the QCSchema file.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#provenance).
    """

    creator: str
    """The name of the creator of this object."""
    version: str
    """The version of the creator of this object."""
    routine: str
    """The routine that was used to create this object."""


@dataclass
class QCModel(_QCBase):
    """A dataclass to store the mathematical model information used in the original calculation.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#model).
    """

    method: str
    """The method used for the computation of this object."""
    basis: str | QCBasisSet
    """The basis set used during the computation. This can be either a simple string or a full
    :class:`QCBasisSet` specification."""

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
    """The list of atom symbols in this topology."""
    geometry: list[float]
    """The XYZ coordinates (in Bohr units) of the atoms. This is a flat list of three times the
    length of the `symbols` list."""
    schema_name: str
    """The name of this schema. This value is expected to be `qcschema_molecule`."""
    schema_version: int
    """The version of this specific schema."""
    molecular_charge: int = None
    """The overall charge of the molecule."""
    molecular_multiplicity: int = None
    """The overall multiplicity of the molecule."""
    fix_com: bool = None
    """Whether translation of the geometry is allowed (`False`) or not (`True`)."""
    real: list[bool] = None
    """A list indicating whether each atom is real (`True`) or a ghost (`False`). Its length must
    match that of the `symbols` list."""
    connectivity: list[tuple[int, int, int]] = None
    """A list indicating the bonds between the atoms in the molecule. Each item of this list must be
    a tuple of three integers, indicating the first atom index in the bond, the second atom index,
    and finally the order of the bond."""
    fix_orientation: bool = None
    """Whether rotation of the geometry is allowed (`False`) or not (`True`)."""
    atom_labels: list[str] = None
    """A list of user-provided information for each atom. Its length must match that of the
    `symbols` list."""
    fragment_multiplicities: list[int] = None
    """The list of multiplicities associated with each fragment."""
    fix_symmetry: str = None
    """The maximal point group symmetry at which the `geometry` should be treated."""
    fragment_charges: list[float] = None
    """The list of charges associated with each fragment."""
    mass_numbers: list[int] = None
    """The mass numbers of all atoms. If it is an unknown isotope, the value should be -1. Its
    length must match that of the `symbols` list."""
    name: str = None
    """The (user-given) name of the molecule."""
    masses: list[float] = None
    """The masses (in atomic units) of all atoms. Canonical weights are assumed if this is not given
    explicitly."""
    comment: str = None
    """Any additional (user-provided) comment."""
    provenance: QCProvenance = None
    """An instance of :class:`QCProvenance`."""
    fragments: list[tuple[int, ...]] = None
    """The list of fragments. Each item of this list must be a tuple of integers with variable
    length (greater than 1). The first number indicates the fragment index, all following numbers
    refer to the (0-indexed) atom indices that constitute this fragment."""
    atomic_numbers: list[int] = None
    """The atomic numbers of all atoms, indicating their nuclear charge. Its length must match that
    of the `symbols` list."""

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
    """The angular momenta of this electron shell as a list of integers."""
    harmonic_type: str
    """The type of this shell."""
    exponents: list[float | str]
    """The exponents of this contracted shell. The official spec stores these values as strings."""
    coefficients: list[list[float | str]]
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
    angular_momentum: list[int]
    """The angular momenta of this potential as a list of integers."""
    r_exponents: list[int]
    """The exponents of the `r` term."""
    gaussian_exponents: list[float | str]
    """The exponents of the gaussian terms. The official spec stores these values as strings."""
    coefficients: list[list[float | str]]
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

    electron_shells: list[QCElectronShell] = None
    """The list of electronic shells for this element."""
    ecp_electrons: int = None
    """The number of electrons replaced by an ECP."""
    ecp_potentials: list[QCECPPotential] = None
    """The list of effective core potentials for this element."""

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
    """A dictionary mapping the keys provided by `atom_map` to their basis center data."""
    atom_map: list[str]
    """The list of atomic kinds, indicating the keys used to store the basis in `center_data`."""
    name: str
    """The name of the basis set."""
    schema_version: int = None
    """The version of this specific schema."""
    schema_name: str = None
    """The name of this schema. This value is expected to be `qcschema_basis`."""
    description: str = None
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


@dataclass
class QCProperties(_QCBase):
    """A dataclass to store the computed properties of the original calculation.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_props.html#properties-schema).
    """

    calcinfo_nbasis: int = None
    """The number of basis functions in the computation."""
    calcinfo_nmo: int = None
    """The number of molecular orbitals in the computation."""
    calcinfo_nalpha: int = None
    """The number of alpha-spin electrons in the computation."""
    calcinfo_nbeta: int = None
    """The number of beta-spin electrons in the computation."""
    calcinfo_natom: int = None
    """The number of atoms in the computation."""
    return_energy: float = None
    """The returned energy of the computation. When :attr:`QCSchemaInput.driver` is `energy`, this
    value is identical to :attr:`QCSchema.return_result`."""

    scf_one_electron_energy: float = None
    """The one-electron energy contribution to the total SCF energy."""
    scf_two_electron_energy: float = None
    """The two-electron energy contribution to the total SCF energy."""
    nuclear_repulsion_energy: float = None
    """The nuclear repulsion energy contribution to the total SCF energy."""
    scf_vv10_energy: float = None
    """The VV10 functional energy contribution to the total SCF energy."""
    scf_xc_energy: float = None
    """The XC functional energy contribution to the total SCF energy."""
    scf_dispersion_correction_energy: float = None
    """The dispersion correction appended to the underlying functional in a DFT-D method."""
    scf_dipole_moment: tuple[float, float, float] = None
    """The total SCF X, Y, and Z dipole components."""
    scf_total_energy: float = None
    """The total SCF energy."""
    scf_iterations: int = None
    """The number of SCF iterations taken during the computation."""

    mp2_same_spin_correlation_energy: float = None
    """The MP2 doubles correlation energy contribution from same-spin (e.g. triplet) correlations,
    without any user scaling."""
    mp2_opposite_spin_correlation_energy: float = None
    """The MP2 doubles correlation energy contribution from opposite-spin (e.g. singlet)
    correlations, without any user scaling."""
    mp2_singles_energy: float = None
    """The MP2 singles correlation energy. This value is `0.0` except in ROHF."""
    mp2_doubles_energy: float = None
    """The total MP2 doubles correlation energy."""
    mp2_correlation_energy: float = None
    """The total MP2 correlation energy."""
    mp2_total_energy: float = None
    """The total MP2 energy (i.e. the sum of the SCF energy and MP2 correlation energy)."""
    mp2_dipole_moment: tuple[float, float, float] = None
    """The total MP2 X, Y, and Z dipole components."""

    ccsd_same_spin_correlation_energy: float = None
    """The CCSD doubles correlation energy contribution from same-spin (e.g. triplet) correlations,
    without any user scaling."""
    ccsd_opposite_spin_correlation_energy: float = None
    """The CCSD doubles correlation energy contribution from opposite-spin (e.g. singlet)
    correlations, without any user scaling."""
    ccsd_singles_energy: float = None
    """The CCSD singles correlation energy. This value is `0.0` except in ROHF."""
    ccsd_doubles_energy: float = None
    """The total CCSD doubles correlation energy."""
    ccsd_correlation_energy: float = None
    """The total CCSD correlation energy."""
    ccsd_total_energy: float = None
    """The total CCSD energy (i.e. the sum of the SCF energy and CCSD correlation energy)."""
    ccsd_prt_pr_correlation_energy: float = None
    """The total CCSD(T) correlation energy."""
    ccsd_prt_pr_total_energy: float = None
    """The total CCSD(T) energy (i.e. the sum of the SCF energy and CCSD(T) correlation energy)."""
    ccsdt_correlation_energy: float = None
    """The total CCSDT correlation energy."""
    ccsdt_total_energy: float = None
    """The total CCSDT energy (i.e. the sum of the SCF energy and CCSDT correlation energy)."""
    ccsdtq_correlation_energy: float = None
    """The total CCSDTQ correlation energy."""
    ccsdtq_total_energy: float = None
    """The total CCSDTQ energy (i.e. the sum of the SCF energy and CCSDTQ correlation energy)."""
    ccsd_dipole_moment: tuple[float, float, float] = None
    """The total CCSD X, Y, and Z dipole components."""
    ccsd_prt_pr_dipole_moment: tuple[float, float, float] = None
    """The total CCSD(T) X, Y, and Z dipole components."""
    ccsdt_dipole_moment: tuple[float, float, float] = None
    """The total CCSDT X, Y, and Z dipole components."""
    ccsdtq_dipole_moment: tuple[float, float, float] = None
    """The total CCSDTQ X, Y, and Z dipole components."""
    ccsd_iterations: int = None
    """The number of CCSD iterations taken during the computation."""
    ccsdt_iterations: int = None
    """The number of CCSDT iterations taken during the computation."""
    ccsdtq_iterations: int = None
    """The number of CCSDTQ iterations taken during the computation."""


@dataclass
class QCWavefunction(_QCBase):
    """A dataclass to store any additional computed wavefunction properties.

    Matrix quantities are stored as flat, column-major arrays.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_wf.html#wavefunction-schema).
    """

    basis: QCBasisSet
    """An instance of :class:`QCBasisSet`."""

    orbitals_a: str = None
    """The name of the alpha-spin orbitals in the AO basis."""
    orbitals_b: str = None
    """The name of the beta-spin orbitals in the AO basis."""
    density_a: str = None
    """The name of the alpha-spin density in the AO basis."""
    density_b: str = None
    """The name of the beta-spin density in the AO basis."""
    density_mo_a: str = None
    """The name of the alpha-spin density in the MO basis."""
    density_mo_b: str = None
    """The name of the beta-spin density in the MO basis."""
    fock_a: str = None
    """The name of the alpha-spin Fock matrix in the AO basis."""
    fock_b: str = None
    """The name of the beta-spin Fock matrix in the AO basis."""
    fock_mo_a: str = None
    """The name of the alpha-spin Fock matrix in the MO basis."""
    fock_mo_b: str = None
    """The name of the beta-spin Fock matrix in the MO basis."""
    eigenvalues_a: str = None
    """The name of the alpha-spin orbital eigenvalues."""
    eigenvalues_b: str = None
    """The name of the beta-spin orbital eigenvalues."""
    occupations_a: str = None
    """The name of the alpha-spin orbital occupations."""
    occupations_b: str = None
    """The name of the beta-spin orbital occupations."""
    eri: str = None
    """The name of the electron-repulsion integrals in the AO basis."""
    eri_mo_aa: str = None
    """The name of the alpha-alpha electron-repulsion integrals in the MO basis."""
    eri_mo_ab: str = None
    """The name of the alpha-beta electron-repulsion integrals in the MO basis."""
    eri_mo_ba: str = None
    """The name of the beta-alpha electron-repulsion integrals in the MO basis."""
    eri_mo_bb: str = None
    """The name of the beta-beta electron-repulsion integrals in the MO basis."""

    scf_orbitals_a: list[float] = None
    """The SCF alpha-spin orbitals in the AO basis."""
    scf_orbitals_b: list[float] = None
    """The SCF beta-spin orbitals in the AO basis."""
    scf_density_a: list[float] = None
    """The SCF alpha-spin density in the AO basis."""
    scf_density_b: list[float] = None
    """The SCF beta-spin density in the AO basis."""
    scf_density_mo_a: list[float] = None
    """The SCF alpha-spin density in the MO basis."""
    scf_density_mo_b: list[float] = None
    """The SCF beta-spin density in the MO basis."""
    scf_fock_a: list[float] = None
    """The SCF alpha-spin Fock matrix in the AO basis."""
    scf_fock_b: list[float] = None
    """The SCF beta-spin Fock matrix in the AO basis."""
    scf_fock_mo_a: list[float] = None
    """The SCF alpha-spin Fock matrix in the MO basis."""
    scf_fock_mo_b: list[float] = None
    """The SCF beta-spin Fock matrix in the MO basis."""
    scf_coulomb_a: list[float] = None
    """The SCF alpha-spin Coulomb matrix in the AO basis."""
    scf_coulomb_b: list[float] = None
    """The SCF beta-spin Coulomb matrix in the AO basis."""
    scf_exchange_a: list[float] = None
    """The SCF alpha-spin Exchange matrix in the AO basis."""
    scf_exchange_b: list[float] = None
    """The SCF beta-spin Exchange matrix in the AO basis."""
    scf_eigenvalues_a: list[float] = None
    """The SCF alpha-spin orbital eigenvalues."""
    scf_eigenvalues_b: list[float] = None
    """The SCF beta-spin orbital eigenvalues."""
    scf_occupations_a: list[float] = None
    """The SCF alpha-spin orbital occupations."""
    scf_occupations_b: list[float] = None
    """The SCF beta-spin orbital occupations."""
    scf_eri: str = None
    """The SCF electron-repulsion integrals in the AO basis."""
    scf_eri_mo_aa: str = None
    """The SCF alpha-alpha electron-repulsion integrals in the MO basis."""
    scf_eri_mo_ab: str = None
    """The SCF alpha-beta electron-repulsion integrals in the MO basis."""
    scf_eri_mo_ba: str = None
    """The SCF beta-alpha electron-repulsion integrals in the MO basis."""
    scf_eri_mo_bb: str = None
    """The SCF beta-beta electron-repulsion integrals in the MO basis."""

    localized_orbitals_a: list[float] = None
    """The localized alpha-spin orbitals. All `nmo` orbitals are included, even if only a subset
    were localized."""
    localized_orbitals_b: list[float] = None
    """The localized beta-spin orbitals. All `nmo` orbitals are included, even if only a subset were
    localized."""
    localized_fock_a: list[float] = None
    """The alpha-spin Fock matrix in the localized basis. All `nmo` orbitals are included, even if
    only a subset were localized."""
    localized_fock_b: list[float] = None
    """The beta-spin Fock matrix in the localized basis. All `nmo` orbitals are included, even if
    only a subset were localized."""

    h_core_a: list[float] = None
    """The alpha-spin core (one-electron) Hamiltonian matrix in the AO basis."""
    h_core_b: list[float] = None
    """The beta-spin core (one-electron) Hamiltonian matrix in the AO basis."""
    h_effective_a: list[float] = None
    """The effective alpha-spin core (one-electron) Hamiltonian matrix in the AO basis."""
    h_effective_b: list[float] = None
    """The effective beta-spin core (one-electron) Hamiltonian matrix in the AO basis."""

    restricted: bool = None
    """Whether the computation used restricted spin orbitals."""

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
    """The name of this schema. This value is expected to be `qcschema` or `qc_schema`."""
    schema_version: int
    """The version of this specific schema."""
    molecule: QCTopology
    """An instance of :class:`QCTopology`."""
    driver: str
    """The type of computation. Example values are `energy`, `gradient`, and `hessian`."""
    model: QCModel
    """An instance of :class:`QCModel`."""
    keywords: dict[str, Any]
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


@dataclass
class QCSchema(QCSchemaInput):
    """The full QCSchema as a dataclass.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#output-components).
    """

    provenance: QCProvenance
    """An instance of :class:`QCProvenance`."""
    return_result: float | list[float]
    """The primary result of the computation. Its value depends on the type of computation (see also
    `driver`)."""
    success: bool
    """Whether the computation was successful."""
    properties: QCProperties
    """An instance of :class:`QCProperties`."""
    error: QCError = None
    """An instance of :class:`QCError` if the computation was not successful (`success = False`)."""
    wavefunction: QCWavefunction = None
    """An instance of :class:`QCWavefunction`."""

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
