# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" HDF5 Driver """

import logging
import os
import warnings

from ..qmolecule import QMolecule
from ..fermionic_driver import FermionicDriver

logger = logging.getLogger(__name__)


class HDF5Driver(FermionicDriver):
    """**DEPRECATED** Qiskit Nature driver for reading an HDF5 file.

    The HDF5 file is as saved from
    a :class:`~qiskit_nature.drivers.QMolecule` instance.
    """

    def __init__(self, hdf5_input: str = "molecule.hdf5") -> None:
        """
        Args:
            hdf5_input: Path to HDF5 file
        """
        super().__init__()
        warnings.warn(
            "This HDF5Driver is deprecated as of 0.2.0, "
            "and will be removed no earlier than 3 months after the release. "
            "You should use the qiskit_nature.drivers.second_quantization.hdf5d "
            "HDF5Driver as a direct replacement instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._hdf5_input = hdf5_input
        self._work_path = None

    @property
    def work_path(self):
        """Returns work path."""
        return self._work_path

    @work_path.setter
    def work_path(self, new_work_path):
        """Sets work path."""
        self._work_path = new_work_path

    def run(self) -> QMolecule:
        """
        Runs driver to produce a QMolecule output.

        Returns:
            A QMolecule containing the molecular data.

        Raises:
            LookupError: file not found.
        """
        hdf5_file = self._hdf5_input
        if self.work_path is not None and not os.path.isabs(hdf5_file):
            hdf5_file = os.path.abspath(os.path.join(self.work_path, hdf5_file))

        if not os.path.isfile(hdf5_file):
            raise LookupError("HDF5 file not found: {}".format(hdf5_file))

        molecule = QMolecule(hdf5_file)
        molecule.load()
        return molecule
