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

"""The QCSchema (output) dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

# Sphinx is somehow unable to resolve this forward reference which gets inherited from QCSchemaInput
# Thus, we import it here manually to ensure the documentation can be built
from typing import Mapping  # pylint: disable=unused-import

import h5py

import numpy as np

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

    @classmethod
    def from_legacy_hdf5(cls, legacy_hdf5: str | Path) -> QCSchema:
        # pylint: disable=line-too-long
        """Constructs a schema object from a legacy HDF5 object.

        .. warning::
            This method will **not** produce a fully valid QCSchema object!
            It will contain all the data needed by Qiskit Nature to continue with any computation,
            but the file might not be usable as input to other codes that take a QCSchema as an
            input, because some of the required data is simply not available in the legacy HDF5
            files.

            In particular, the following attributes are known to be invalid:

            - :attr:`.QCSchemaInput.schema_version`
            - :attr:`.QCModel.method`
            - :attr:`.QCModel.basis`
            - :attr:`.QCWavefunction.basis`

        Args:
            legacy_hdf5: the path to the legacy HDF5 file.

        Raises:
            ValueError: if the provided legacy HDF5 file contains anything other than a
            :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicStructureDriverResult`

        Returns:
            An instance of the schema object.
        """
        # values which are already provided here, ensure that all required init arguments exist,
        # once it comes to the construction of the QCSchema object
        topology_data: dict[str, Any] = {
            "symbols": [],
            "geometry": [],
        }
        properties_data: dict[str, Any] = {}
        provenance_data: dict[str, Any] = {
            "creator": "Qiskit Nature",
            "version": __version__,
            "routine": f"{cls.__module__}.{cls.__name__}.from_legacy_hdf5",
        }
        wavefunction_data: dict[str, Any] = {}
        return_result: float | None = None

        root_name = "ElectronicStructureDriverResult"

        with h5py.File(legacy_hdf5, "r") as file:
            if root_name not in file.keys():
                raise ValueError(
                    "A QCSchema can only be constructed from a legacy "
                    f"ElectronicStructureDriverResult, not from a {set(file.keys()).pop()}.\n"
                    "You may want to look for other formats available in "
                    "qiskit_nature.second_q.formats to see if one of those matches your desired "
                    "object type."
                )

            driver_metadata = file[root_name].get("DriverMetadata", None)
            if driver_metadata is not None:
                provenance_data["creator"] = driver_metadata.attrs["program"]
                provenance_data["version"] = driver_metadata.attrs["version"]
                provenance_data["routine"] = driver_metadata.attrs["config"]

            molecule = file[root_name].get("Molecule", None)
            if molecule is not None:
                symbols: list[str] = []
                geometry: list[float] = []

                for atom in molecule["geometry"].values():
                    symbols.append(atom.attrs["symbol"])
                    geometry.extend(atom[...].tolist())

                properties_data["calcinfo_natom"] = len(symbols)
                topology_data["symbols"] = symbols
                topology_data["geometry"] = geometry

                if "multiplicity" in molecule.attrs.keys():
                    topology_data["molecular_multiplicity"] = int(molecule.attrs["multiplicity"])
                if "charge" in molecule.attrs.keys():
                    topology_data["molecular_charge"] = int(molecule.attrs["charge"])
                if "masses" in molecule.keys():
                    topology_data["masses"] = list(molecule["masses"])

            basis_transform = file[root_name].get("ElectronicBasisTransform", None)
            if basis_transform is not None:
                if (
                    basis_transform.attrs["initial_basis"] == "AO"
                    and basis_transform.attrs["final_basis"] == "MO"
                ):
                    orbitals_a = np.asarray(basis_transform["Alpha coefficients"][...])
                    nao, nmo = orbitals_a.shape
                    properties_data["calcinfo_nbasis"] = nao
                    properties_data["calcinfo_nmo"] = nmo

                    wavefunction_data["orbitals_a"] = "scf_orbitals_a"
                    wavefunction_data["scf_orbitals_a"] = orbitals_a.ravel().tolist()
                    if "Beta coefficients" in basis_transform.keys():
                        wavefunction_data["orbitals_b"] = "scf_orbitals_b"
                        wavefunction_data["scf_orbitals_b"] = (
                            np.asarray(basis_transform["Beta coefficients"][...]).ravel().tolist()
                        )

            particle_number = file[root_name].get("ParticleNumber", None)
            if particle_number is not None:
                if properties_data.get("calcinfo_nmo", None) is None:
                    # we might have gotten this number from the basis transform already
                    properties_data["calcinfo_nmo"] = (
                        int(particle_number.attrs["num_spin_orbitals"]) // 2
                    )
                properties_data["calcinfo_nalpha"] = int(particle_number.attrs["num_alpha"])
                properties_data["calcinfo_nbeta"] = int(particle_number.attrs["num_beta"])
                wavefunction_data["occupations_a"] = "scf_occupations_a"
                wavefunction_data["occupations_b"] = "scf_occupations_b"
                wavefunction_data["scf_occupations_a"] = list(
                    particle_number["occupation_alpha"][...]
                )
                wavefunction_data["scf_occupations_b"] = list(
                    particle_number["occupation_beta"][...]
                )

            electronic_energy = file[root_name].get("ElectronicEnergy", None)
            if electronic_energy is not None:
                if "reference_energy" in electronic_energy.attrs.keys():
                    return_result = float(electronic_energy.attrs["reference_energy"])
                    properties_data["return_energy"] = return_result
                if "nuclear_repulsion_energy" in electronic_energy.attrs.keys():
                    properties_data["nuclear_repulsion_energy"] = float(
                        electronic_energy.attrs["nuclear_repulsion_energy"]
                    )

                orbital_energies = electronic_energy.attrs.get("orbital_energies", None)
                if orbital_energies is not None:
                    orbital_energies = np.asarray(orbital_energies)

                    wavefunction_data["eigenvalues_a"] = "scf_eigenvalues_a"

                    if len(orbital_energies.shape) == 2:
                        wavefunction_data["eigenvalues_b"] = "scf_eigenvalues_b"
                        wavefunction_data["scf_eigenvalues_a"] = (
                            orbital_energies[0].ravel().tolist()
                        )
                        wavefunction_data["scf_eigenvalues_b"] = (
                            orbital_energies[1].ravel().tolist()
                        )
                    else:
                        wavefunction_data["scf_eigenvalues_a"] = orbital_energies.ravel().tolist()

                electronic_integrals = electronic_energy["electronic_integrals"]

                def _extract_electronic_integral(
                    basis: str,
                    nbody: str,
                    spin: str,
                    qcschema_key: str,
                ):
                    basis_integrals = electronic_integrals.get(basis, None)
                    if basis_integrals is None:
                        return
                    nbody_integrals = basis_integrals.get(f"{nbody}BodyElectronicIntegrals", None)
                    if nbody_integrals is None:
                        return
                    spin_integrals = nbody_integrals.get(spin, None)
                    if spin_integrals is None:
                        return
                    wavefunction_data[qcschema_key] = f"scf_{qcschema_key}"
                    wavefunction_data[f"scf_{qcschema_key}"] = (
                        np.asarray(spin_integrals).ravel().tolist()
                    )

                _extract_electronic_integral("AO", "One", "Alpha", "fock_a")
                _extract_electronic_integral("AO", "One", "Beta", "fock_b")
                _extract_electronic_integral("AO", "Two", "Alpha-Alpha", "eri")

                _extract_electronic_integral("MO", "One", "Alpha", "fock_mo_a")
                _extract_electronic_integral("MO", "One", "Beta", "fock_mo_b")
                _extract_electronic_integral("MO", "Two", "Alpha-Alpha", "eri_mo_aa")
                _extract_electronic_integral("MO", "Two", "Beta-Alpha", "eri_mo_ba")
                _extract_electronic_integral("MO", "Two", "Beta-Beta", "eri_mo_bb")
                _extract_electronic_integral("MO", "Two", "Alpha-Beta", "eri_mo_ab")

            return cls(
                schema_name="qcschema",
                schema_version=-1,  # this is meant to indicate, that this is not a fully valid file
                molecule=QCTopology(
                    schema_name="qcschema_molecule",
                    schema_version=2,
                    **topology_data,
                ),
                driver="energy",
                model=QCModel(
                    method="?",
                    basis="?",
                ),
                keywords={},
                provenance=QCProvenance(**provenance_data),
                return_result=return_result,
                success=True,
                properties=QCProperties(**properties_data),
                wavefunction=QCWavefunction(
                    basis="?",  # this is technically invalid, but the data simply does not exist
                    **wavefunction_data,
                ),
            )
