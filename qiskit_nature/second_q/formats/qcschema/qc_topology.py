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

"""The QCSchema topology dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import h5py

from .qc_base import _QCBase
from .qc_provenance import QCProvenance


@dataclass
class QCTopology(_QCBase):
    """A dataclass to store the topological information of the physical system.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#topology).
    """

    symbols: Sequence[str]
    """The list of atom symbols in this topology."""
    geometry: Sequence[float]
    """The XYZ coordinates (in Bohr units) of the atoms. This is a flat list of three times the
    length of the `symbols` list."""
    schema_name: str
    """The name of this schema. This value is expected to be `qcschema_molecule`."""
    schema_version: int
    """The version of this specific schema."""
    molecular_charge: int | None = None
    """The overall charge of the molecule."""
    molecular_multiplicity: int | None = None
    """The overall multiplicity of the molecule."""
    fix_com: bool | None = None
    """Whether translation of the geometry is allowed (`False`) or not (`True`)."""
    real: Sequence[bool] | None = None
    """A list indicating whether each atom is real (`True`) or a ghost (`False`). Its length must
    match that of the `symbols` list."""
    connectivity: Sequence[tuple[int, int, int]] | None = None
    """A list indicating the bonds between the atoms in the molecule. Each item of this list must be
    a tuple of three integers, indicating the first atom index in the bond, the second atom index,
    and finally the order of the bond."""
    fix_orientation: bool | None = None
    """Whether rotation of the geometry is allowed (`False`) or not (`True`)."""
    atom_labels: Sequence[str] | None = None
    """A list of user-provided information for each atom. Its length must match that of the
    `symbols` list."""
    fragment_multiplicities: Sequence[int] | None = None
    """The list of multiplicities associated with each fragment."""
    fix_symmetry: str | None = None
    """The maximal point group symmetry at which the `geometry` should be treated."""
    fragment_charges: Sequence[float] | None = None
    """The list of charges associated with each fragment."""
    mass_numbers: Sequence[int] | None = None
    """The mass numbers of all atoms. If it is an unknown isotope, the value should be -1. Its
    length must match that of the `symbols` list."""
    name: str | None = None
    """The (user-given) name of the molecule."""
    masses: Sequence[float] | None = None
    """The masses (in atomic units) of all atoms. Canonical weights are assumed if this is not given
    explicitly."""
    comment: str | None = None
    """Any additional (user-provided) comment."""
    provenance: QCProvenance | None = None
    """An instance of :class:`QCProvenance`."""
    fragments: Sequence[tuple[int, ...]] | None = None
    """The list of fragments. Each item of this list must be a tuple of integers with variable
    length (greater than 1). The first number indicates the fragment index, all following numbers
    refer to the (0-indexed) atom indices that constitute this fragment."""
    atomic_numbers: Sequence[int] | None = None
    """The atomic numbers of all atoms, indicating their nuclear charge. Its length must match that
    of the `symbols` list."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCTopology:
        provenance: QCProvenance | None = None
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
