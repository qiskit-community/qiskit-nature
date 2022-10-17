# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
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

import numpy as np

from qiskit_nature.second_q.formats.qcschema import (
    QCModel,
    QCProperties,
    QCProvenance,
    QCSchema,
    QCTopology,
    QCWavefunction,
)
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
    mo_coeff: np.ndarray | None = None
    mo_coeff_b: np.ndarray | None = None
    mo_energy: np.ndarray | None = None
    mo_energy_b: np.ndarray | None = None
    mo_occ: np.ndarray | None = None
    mo_occ_b: np.ndarray | None = None
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
    def to_qcschema(self) -> QCSchema:
        """Extracts all available information after the driver was run into a :class:`.QCSchema`
        object.

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
    def _to_qcschema(data: _QCSchemaData) -> QCSchema:
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

        def format_np_array(arr):
            return arr.ravel().tolist()

        wavefunction = QCWavefunction(basis=data.basis)
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
