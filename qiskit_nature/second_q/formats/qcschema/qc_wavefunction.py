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

"""The QCSchema wavefunction dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import h5py

from .qc_base import _QCBase
from .qc_basis_set import QCBasisSet


@dataclass
class QCWavefunction(_QCBase):
    """A dataclass to store any additional computed wavefunction properties.

    Matrix quantities are stored as flat, column-major arrays.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_wf.html#wavefunction-schema).
    """

    basis: QCBasisSet
    """An instance of :class:`QCBasisSet`."""

    orbitals_a: str | None = None
    """The name of the alpha-spin orbitals in the AO basis."""
    orbitals_b: str | None = None
    """The name of the beta-spin orbitals in the AO basis."""
    density_a: str | None = None
    """The name of the alpha-spin density in the AO basis."""
    density_b: str | None = None
    """The name of the beta-spin density in the AO basis."""
    density_mo_a: str | None = None
    """The name of the alpha-spin density in the MO basis."""
    density_mo_b: str | None = None
    """The name of the beta-spin density in the MO basis."""
    fock_a: str | None = None
    """The name of the alpha-spin Fock matrix in the AO basis."""
    fock_b: str | None = None
    """The name of the beta-spin Fock matrix in the AO basis."""
    fock_mo_a: str | None = None
    """The name of the alpha-spin Fock matrix in the MO basis."""
    fock_mo_b: str | None = None
    """The name of the beta-spin Fock matrix in the MO basis."""
    eigenvalues_a: str | None = None
    """The name of the alpha-spin orbital eigenvalues."""
    eigenvalues_b: str | None = None
    """The name of the beta-spin orbital eigenvalues."""
    occupations_a: str | None = None
    """The name of the alpha-spin orbital occupations."""
    occupations_b: str | None = None
    """The name of the beta-spin orbital occupations."""
    eri: str | None = None
    """The name of the electron-repulsion integrals in the AO basis."""
    eri_mo_aa: str | None = None
    """The name of the alpha-alpha electron-repulsion integrals in the MO basis."""
    eri_mo_ab: str | None = None
    """The name of the alpha-beta electron-repulsion integrals in the MO basis."""
    eri_mo_ba: str | None = None
    """The name of the beta-alpha electron-repulsion integrals in the MO basis."""
    eri_mo_bb: str | None = None
    """The name of the beta-beta electron-repulsion integrals in the MO basis."""

    scf_orbitals_a: Sequence[float] | None = None
    """The SCF alpha-spin orbitals in the AO basis."""
    scf_orbitals_b: Sequence[float] | None = None
    """The SCF beta-spin orbitals in the AO basis."""
    scf_density_a: Sequence[float] | None = None
    """The SCF alpha-spin density in the AO basis."""
    scf_density_b: Sequence[float] | None = None
    """The SCF beta-spin density in the AO basis."""
    scf_density_mo_a: Sequence[float] | None = None
    """The SCF alpha-spin density in the MO basis."""
    scf_density_mo_b: Sequence[float] | None = None
    """The SCF beta-spin density in the MO basis."""
    scf_fock_a: Sequence[float] | None = None
    """The SCF alpha-spin Fock matrix in the AO basis."""
    scf_fock_b: Sequence[float] | None = None
    """The SCF beta-spin Fock matrix in the AO basis."""
    scf_fock_mo_a: Sequence[float] | None = None
    """The SCF alpha-spin Fock matrix in the MO basis."""
    scf_fock_mo_b: Sequence[float] | None = None
    """The SCF beta-spin Fock matrix in the MO basis."""
    scf_coulomb_a: Sequence[float] | None = None
    """The SCF alpha-spin Coulomb matrix in the AO basis."""
    scf_coulomb_b: Sequence[float] | None = None
    """The SCF beta-spin Coulomb matrix in the AO basis."""
    scf_exchange_a: Sequence[float] | None = None
    """The SCF alpha-spin Exchange matrix in the AO basis."""
    scf_exchange_b: Sequence[float] | None = None
    """The SCF beta-spin Exchange matrix in the AO basis."""
    scf_eigenvalues_a: Sequence[float] | None = None
    """The SCF alpha-spin orbital eigenvalues."""
    scf_eigenvalues_b: Sequence[float] | None = None
    """The SCF beta-spin orbital eigenvalues."""
    scf_occupations_a: Sequence[float] | None = None
    """The SCF alpha-spin orbital occupations."""
    scf_occupations_b: Sequence[float] | None = None
    """The SCF beta-spin orbital occupations."""
    scf_eri: str | None = None
    """The SCF electron-repulsion integrals in the AO basis."""
    scf_eri_mo_aa: str | None = None
    """The SCF alpha-alpha electron-repulsion integrals in the MO basis."""
    scf_eri_mo_ab: str | None = None
    """The SCF alpha-beta electron-repulsion integrals in the MO basis."""
    scf_eri_mo_ba: str | None = None
    """The SCF beta-alpha electron-repulsion integrals in the MO basis."""
    scf_eri_mo_bb: str | None = None
    """The SCF beta-beta electron-repulsion integrals in the MO basis."""

    localized_orbitals_a: Sequence[float] | None = None
    """The localized alpha-spin orbitals. All `nmo` orbitals are included, even if only a subset
    were localized."""
    localized_orbitals_b: Sequence[float] | None = None
    """The localized beta-spin orbitals. All `nmo` orbitals are included, even if only a subset were
    localized."""
    localized_fock_a: Sequence[float] | None = None
    """The alpha-spin Fock matrix in the localized basis. All `nmo` orbitals are included, even if
    only a subset were localized."""
    localized_fock_b: Sequence[float] | None = None
    """The beta-spin Fock matrix in the localized basis. All `nmo` orbitals are included, even if
    only a subset were localized."""

    h_core_a: Sequence[float] | None = None
    """The alpha-spin core (one-electron) Hamiltonian matrix in the AO basis."""
    h_core_b: Sequence[float] | None = None
    """The beta-spin core (one-electron) Hamiltonian matrix in the AO basis."""
    h_effective_a: Sequence[float] | None = None
    """The effective alpha-spin core (one-electron) Hamiltonian matrix in the AO basis."""
    h_effective_b: Sequence[float] | None = None
    """The effective beta-spin core (one-electron) Hamiltonian matrix in the AO basis."""

    restricted: bool | None = None
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
