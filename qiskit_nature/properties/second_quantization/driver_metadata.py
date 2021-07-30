# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The DriverMetadata class."""

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
