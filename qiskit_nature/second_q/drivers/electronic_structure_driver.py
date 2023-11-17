# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for electronic structure driver modules.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from pathlib import Path
import numpy as np
import h5py

from qiskit_nature.second_q.formats.qcschema import (
    QCModel,
    QCProperties,
    QCProvenance,
    QCSchema,
    QCTopology,
    QCWavefunction,
)
from qiskit_nature.second_q.operators import Tensor
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem

from .base_driver import BaseDriver


@dataclass
class _QCSchemaData:
    """An internal data container to simplify the construction of QCSchema objects."""

    hij: np.ndarray | None = None
    hij_b: np.ndarray | None = None
    eri: np.ndarray | None = None
    hij_mo: np.ndarray | None = None
    hij_mo_b: np.ndarray | None = None
    eri_mo: np.ndarray | None = None
    eri_mo_ba: np.ndarray | None = None
    eri_mo_bb: np.ndarray | None = None
    e_nuc: float | None = None
    e_ref: float | None = None
    overlap: np.ndarray | None = None
    mo_coeff: np.ndarray | None = None
    mo_coeff_b: np.ndarray | None = None
    mo_energy: np.ndarray | None = None
    mo_energy_b: np.ndarray | None = None
    mo_occ: np.ndarray | None = None
    mo_occ_b: np.ndarray | None = None
    dip_x: np.ndarray | None = None
    dip_y: np.ndarray | None = None
    dip_z: np.ndarray | None = None
    dip_mo_x_a: np.ndarray | None = None
    dip_mo_y_a: np.ndarray | None = None
    dip_mo_z_a: np.ndarray | None = None
    dip_mo_x_b: np.ndarray | None = None
    dip_mo_y_b: np.ndarray | None = None
    dip_mo_z_b: np.ndarray | None = None
    dip_nuc: tuple[float, float, float] | None = None
    dip_ref: tuple[float, float, float] | None = None
    symbols: Sequence[str] | None = None
    coords: Sequence[float] | None = None
    multiplicity: int | None = None
    charge: int | None = None
    masses: Sequence[float] | None = None
    method: str | None = None
    basis: str | None = None
    creator: str | None = None
    version: str | None = None
    routine: str | None = None
    nbasis: int | None = None
    nmo: int | None = None
    nalpha: int | None = None
    nbeta: int | None = None
    keywords: dict[str, Any] | None = None

    def to_hdf5(self, h5py_data: str | Path | h5py.Group) -> None:
        """Converts the schema data object to HDF5.

        Args:
            h5py_data: the h5py group into which to store the object.
        """
        if isinstance(h5py_data, h5py.Group):
            self._to_hdf5(h5py_data)
            return

        with h5py.File(h5py_data, "w") as file:
            self._to_hdf5(file)

    def _to_hdf5(self, group: h5py.Group) -> None:
        """This internal method deals with actually converting the schema data object to HDF5.

        Args:
            group: the h5py group into which to store the object.
        """
        # we use __dict__ here because we do not want the recursive behavior of asdict()
        for key, value in self.__dict__.items():
            if value is None:
                continue

            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            else:
                group.attrs[key] = value

    @classmethod
    def from_hdf5(cls, h5py_data: str | Path | h5py.Group) -> _QCSchemaData:
        """Constructs a schema object from an HDF5 object.

        Args:
            h5py_data: can be either the path to a file or an `h5py.Group`.

        Returns:
            An instance of the schema data object.
        """
        if isinstance(h5py_data, h5py.Group):
            return cls._from_hdf5_group(h5py_data)

        with h5py.File(h5py_data, "r") as file:
            return cls._from_hdf5_group(file)

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> _QCSchemaData:
        """This internal method deals with actually constructing a schema
        data object from an `h5py.Group`.

        Args:
            h5py_group: the actual `h5py.Group`.

        Returns:
            An instance of the schema data object.
        """
        data = dict(h5py_group.attrs.items())
        for key, value in h5py_group.items():
            data[key] = value[...]
        for key, value in data.items():
            if key in {"symbols", "coords", "masses"}:
                data[key] = value.tolist()
            elif key in {"dip_nuc", "dip_ref"}:
                data[key] = (value[0], value[1], value[2])

        return cls(**data)


class MethodType(Enum):
    """MethodType Enum

    The HF-style methods are common names which are likely available everywhere.
    The KS-style methods are not available for all drivers. Please check the specific driver
    documentation for details.
    """

    RHF = "rhf"
    ROHF = "rohf"
    UHF = "uhf"
    RKS = "rks"
    ROKS = "roks"
    UKS = "uks"


class ElectronicStructureDriver(BaseDriver):
    """
    Base class for Qiskit Nature's electronic structure drivers.
    """

    @abstractmethod
    def run(self) -> ElectronicStructureProblem:
        """Returns an :class:`.ElectronicStructureProblem` output as produced by the driver."""
        pass

    @abstractmethod
    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        """Extracts all available information after the driver was run into a :class:`.QCSchema`
        object.

        Args:
            include_dipole: whether or not to include the custom dipole integrals in the QCSchema.

        Returns:
            A :class:`.QCSchema` storing all extracted system data computed by the driver.
        """

    @abstractmethod
    def to_problem(
        self,
        *,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        """Extends the :meth:`to_qcschema` method and translates the :class:`.QCSchema` object to an
        :class:`.ElectronicStructureProblem`.

        Args:
            basis: the :class:`.ElectronicBasis` in which to construct the problem.
            include_dipole: whether or not to include an :class:`.ElectronicDipoleMoment` property
                in the generated problem (if the data is available).

        Returns:
            An :class:`.ElectronicStructureProblem`.
        """

    @staticmethod
    def _to_qcschema(data: _QCSchemaData, *, include_dipole: bool = True) -> QCSchema:
        molecule = QCTopology(
            schema_name="qcschema_molecule",
            schema_version=2,
            symbols=data.symbols,
            geometry=data.coords,
            molecular_charge=data.charge,
            molecular_multiplicity=data.multiplicity,
            masses=data.masses,
        )

        properties = QCProperties()
        properties.calcinfo_natom = len(data.symbols) if data.symbols is not None else None
        properties.calcinfo_nbasis = data.nbasis
        properties.calcinfo_nmo = data.nmo
        properties.calcinfo_nalpha = data.nalpha
        properties.calcinfo_nbeta = data.nbeta
        properties.return_energy = data.e_ref
        properties.nuclear_repulsion_energy = data.e_nuc
        properties.nuclear_dipole_moment = data.dip_nuc
        properties.scf_dipole_moment = data.dip_ref

        def format_np_array(arr):
            if isinstance(arr, Tensor):
                # NOTE: this also deals with symmetry-reduced integral classes and ensures that
                # they are not automatically unfolded to 1-fold symmetry
                arr = arr.array
            return arr.ravel().tolist()

        wavefunction = QCWavefunction(basis=data.basis)
        if data.overlap is not None:
            wavefunction.overlap = "scf_overlap"
            wavefunction.scf_overlap = format_np_array(data.overlap)
        if data.mo_coeff is not None:
            wavefunction.orbitals_a = "scf_orbitals_a"
            wavefunction.scf_orbitals_a = format_np_array(data.mo_coeff)
        if data.mo_coeff_b is not None:
            wavefunction.orbitals_b = "scf_orbitals_b"
            wavefunction.scf_orbitals_b = format_np_array(data.mo_coeff_b)
        if data.mo_occ is not None:
            wavefunction.occupations_a = "scf_occupations_a"
            wavefunction.scf_occupations_a = format_np_array(data.mo_occ)
        if data.mo_occ_b is not None:
            wavefunction.occupations_b = "scf_occupations_b"
            wavefunction.scf_occupations_b = format_np_array(data.mo_occ_b)
        if data.mo_energy is not None:
            wavefunction.eigenvalues_a = "scf_eigenvalues_a"
            wavefunction.scf_eigenvalues_a = format_np_array(data.mo_energy)
        if data.mo_energy_b is not None:
            wavefunction.eigenvalues_b = "scf_eigenvalues_b"
            wavefunction.scf_eigenvalues_b = format_np_array(data.mo_energy_b)
        if data.hij is not None:
            wavefunction.fock_a = "scf_fock_a"
            wavefunction.scf_fock_a = format_np_array(data.hij)
        if data.hij_b is not None:
            wavefunction.fock_b = "scf_fock_b"
            wavefunction.scf_fock_b = format_np_array(data.hij_b)
        if data.hij_mo is not None:
            wavefunction.fock_mo_a = "scf_fock_mo_a"
            wavefunction.scf_fock_mo_a = format_np_array(data.hij_mo)
        if data.hij_mo_b is not None:
            wavefunction.fock_mo_b = "scf_fock_mo_b"
            wavefunction.scf_fock_mo_b = format_np_array(data.hij_mo_b)
        if data.eri is not None:
            wavefunction.eri = "scf_eri"
            wavefunction.scf_eri = format_np_array(data.eri)
        if data.eri_mo is not None:
            wavefunction.eri_mo_aa = "scf_eri_mo_aa"
            wavefunction.scf_eri_mo_aa = format_np_array(data.eri_mo)
        if data.eri_mo_ba is not None:
            wavefunction.eri_mo_ba = "scf_eri_mo_ba"
            wavefunction.scf_eri_mo_ba = format_np_array(data.eri_mo_ba)
        if data.eri_mo_bb is not None:
            wavefunction.eri_mo_bb = "scf_eri_mo_bb"
            wavefunction.scf_eri_mo_bb = format_np_array(data.eri_mo_bb)
        if include_dipole:
            if data.dip_x is not None:
                wavefunction.dipole_x = "scf_dipole_x"
                wavefunction.scf_dipole_x = format_np_array(data.dip_x)
            if data.dip_y is not None:
                wavefunction.dipole_y = "scf_dipole_y"
                wavefunction.scf_dipole_y = format_np_array(data.dip_y)
            if data.dip_z is not None:
                wavefunction.dipole_z = "scf_dipole_z"
                wavefunction.scf_dipole_z = format_np_array(data.dip_z)
            if data.dip_mo_x_a is not None:
                wavefunction.dipole_mo_x_a = "scf_dipole_mo_x_a"
                wavefunction.scf_dipole_mo_x_a = format_np_array(data.dip_mo_x_a)
            if data.dip_mo_y_a is not None:
                wavefunction.dipole_mo_y_a = "scf_dipole_mo_y_a"
                wavefunction.scf_dipole_mo_y_a = format_np_array(data.dip_mo_y_a)
            if data.dip_mo_z_a is not None:
                wavefunction.dipole_mo_z_a = "scf_dipole_mo_z_a"
                wavefunction.scf_dipole_mo_z_a = format_np_array(data.dip_mo_z_a)
            if data.dip_mo_x_b is not None:
                wavefunction.dipole_mo_x_b = "scf_dipole_mo_x_b"
                wavefunction.scf_dipole_mo_x_b = format_np_array(data.dip_mo_x_b)
            if data.dip_mo_y_b is not None:
                wavefunction.dipole_mo_y_b = "scf_dipole_mo_y_b"
                wavefunction.scf_dipole_mo_y_b = format_np_array(data.dip_mo_y_b)
            if data.dip_mo_z_b is not None:
                wavefunction.dipole_mo_z_b = "scf_dipole_mo_z_b"
                wavefunction.scf_dipole_mo_z_b = format_np_array(data.dip_mo_z_b)

        qcschema = QCSchema(
            schema_name="qcschema",
            schema_version=3,
            molecule=molecule,
            driver="energy",
            model=QCModel(
                method=data.method,
                basis=data.basis,
            ),
            keywords=data.keywords if data.keywords is not None else {},
            provenance=QCProvenance(
                creator=data.creator,
                version=data.version,
                routine=data.routine if data.routine is not None else "",
            ),
            return_result=data.e_ref,
            success=True,
            properties=properties,
            wavefunction=wavefunction,
        )
        return qcschema
