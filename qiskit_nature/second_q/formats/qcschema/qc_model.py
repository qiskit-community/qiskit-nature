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

"""The QCSchema model dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import h5py

from .qc_base import _QCBase
from .qc_basis_set import QCBasisSet


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
