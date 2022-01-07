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

import h5py

from ..property import PseudoProperty


class DriverMetadata(PseudoProperty):
    """A meta-data storage container for driver information."""

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

    def to_hdf5(self, parent: h5py.Group):
        """TODO."""
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["program"] = self.program
        group.attrs["version"] = self.version
        group.attrs["config"] = self.config

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> "DriverMetadata":
        """TODO."""
        return DriverMetadata(
            h5py_group.attrs["program"],
            h5py_group.attrs["version"],
            h5py_group.attrs["config"],
        )
