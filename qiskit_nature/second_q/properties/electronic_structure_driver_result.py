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

"""The ElectronicStructureDriverResult class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py

from qiskit_nature.second_q.operators import FermionicOp

from .second_quantized_property import SecondQuantizedProperty
from .electronic_types import GroupedElectronicProperty

if TYPE_CHECKING:
    from qiskit_nature.second_q.drivers import Molecule


class ElectronicStructureDriverResult(GroupedElectronicProperty):
    """The ElectronicStructureDriverResult class.

    This is a :class:`~qiskit_nature.properties.GroupedProperty` gathering all property objects.
    """

    def __init__(self) -> None:
        """
        Property objects should be added via ``add_property`` rather than via the initializer.
        """
        super().__init__(self.__class__.__name__)

        self.molecule: "Molecule" = None

    def __str__(self) -> str:
        string = [super().__str__()]
        string += [str(self.molecule)]
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)
        self.molecule.to_hdf5(group)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> ElectronicStructureDriverResult:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        grouped_property = GroupedElectronicProperty.from_hdf5(h5py_group)

        ret = ElectronicStructureDriverResult()

        from qiskit_nature.second_q.drivers import Molecule

        for prop in grouped_property:
            if isinstance(prop, Molecule):
                ret.molecule = prop
            else:
                ret.add_property(prop)

        return ret

    def second_q_ops(self) -> dict[str, FermionicOp]:
        """Returns the second quantized operators associated with the properties in this group.

        Returns:
            A `dict` of `FermionicOp` objects.
        """
        ops = {}
        for prop in iter(self):
            if not isinstance(prop, SecondQuantizedProperty):
                continue
            ops.update(prop.second_q_ops())
        return ops
