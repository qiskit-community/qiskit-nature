# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The VibrationalStructureDriverResult class."""

from __future__ import annotations

import h5py

from qiskit_nature.second_q.operators import VibrationalOp

from .bases import VibrationalBasis
from .vibrational_types import GroupedVibrationalProperty


class VibrationalStructureDriverResult(GroupedVibrationalProperty):
    """The VibrationalStructureDriverResult class.

    This is a :class:`~qiskit_nature.properties.GroupedProperty` gathering all property objects.
    """

    def __init__(self) -> None:
        """
        Property objects should be added via ``add_property`` rather than via the initializer.
        """
        super().__init__(self.__class__.__name__)
        self._num_modes: int = None

    @property
    def num_modes(self) -> int:
        """Returns the number of modes."""
        return self._num_modes

    @num_modes.setter
    def num_modes(self, num_modes: int) -> None:
        """Sets the number of modes."""
        self._num_modes = num_modes

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["num_modes"] = self.num_modes

        if self.basis is not None:
            self.basis.to_hdf5(group)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> VibrationalStructureDriverResult:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        grouped_property = GroupedVibrationalProperty.from_hdf5(h5py_group)

        basis: VibrationalBasis = None
        ret = VibrationalStructureDriverResult()
        for prop in grouped_property:
            if isinstance(prop, VibrationalBasis):
                basis = prop
                continue
            ret.add_property(prop)

        ret.num_modes = h5py_group.attrs["num_modes"]

        if basis is not None:
            ret.basis = basis

        return ret

    def second_q_ops(self) -> dict[str, VibrationalOp]:
        """Returns the second quantized operators associated with the properties in this group.

        Returns:
            A `dict` of `VibrationalOp` objects.
        """
        ops = {}
        for prop in iter(self):
            ops.update(prop.second_q_ops())
        return ops
