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

"""The DriverMetadata class."""

from __future__ import annotations

import h5py

from ..property import Property


class DriverMetadata(Property):
    """A meta-data storage container for driver information."""

    _HDF5_ATTR_PROGRAM = "program"
    _HDF5_ATTR_VERSION = "version"
    _HDF5_ATTR_CONFIG = "config"

    def __init__(self, program: str, version: str, config: str) -> None:
        """
        Args:
            program: the name of the classical code run by the driver.
            version: the version of the classical code.
            config: the configuration of the classical code.
        """
        super().__init__(self.__class__.__name__)
        self.program = program
        self.version = version
        self.config = config

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\tProgram: {self.program}"]
        string += [f"\tVersion: {self.version}"]
        string += ["\tConfig:"]
        string += ["\t\t" + s for s in self.config.split("\n")]
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs[DriverMetadata._HDF5_ATTR_PROGRAM] = self.program
        group.attrs[DriverMetadata._HDF5_ATTR_VERSION] = self.version
        group.attrs[DriverMetadata._HDF5_ATTR_CONFIG] = self.config

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> DriverMetadata:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        return DriverMetadata(
            h5py_group.attrs[DriverMetadata._HDF5_ATTR_PROGRAM],
            h5py_group.attrs[DriverMetadata._HDF5_ATTR_VERSION],
            h5py_group.attrs[DriverMetadata._HDF5_ATTR_CONFIG],
        )
