# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
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

import h5py

from qiskit_nature.hdf5 import load_from_hdf5, save_to_hdf5
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult

from ...qmolecule import QMolecule
from ..base_driver import BaseDriver

LOGGER = logging.getLogger(__name__)


class HDF5Driver(BaseDriver):
    """
    Qiskit Nature driver for reading an HDF5 file.

    The HDF5 file is one constructed with :func:`~qiskit_nature.hdf5.save_to_hdf5` or a file
    containing a legacy :class:`~qiskit_nature.drivers.QMolecule` instance.
    """

    def __init__(self, hdf5_input: str = "molecule.hdf5") -> None:
        """
        Args:
            hdf5_input: Path to HDF5 file
        """
        super().__init__()
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

    def _get_path(self) -> str:
        """Returns the absolute path to the HDF5 file.

        Raises:
            LoopupError: file not found.
        """
        hdf5_file = self._hdf5_input
        if self.work_path is not None and not os.path.isabs(hdf5_file):
            hdf5_file = os.path.abspath(os.path.join(self.work_path, hdf5_file))

        if not os.path.isfile(hdf5_file):
            raise LookupError(f"HDF5 file not found: {hdf5_file}")

        return hdf5_file

    def convert(self, replace: bool = True) -> None:
        """Converts a legacy QMolecule HDF5 file into the new Property-framework.

        Args:
            replace: if True, will replace the original HDF5 file. Otherwise `.new` will be used as
                a suffix.

        Raises:
            LookupError: file not found.
        """
        hdf5_file = self._get_path()

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        q_mol = QMolecule(hdf5_file)
        warnings.filterwarnings("default", category=DeprecationWarning)
        q_mol.load()

        new_hdf5_file = hdf5_file if replace else hdf5_file + ".new"

        driver_result = ElectronicStructureDriverResult.from_legacy_driver_result(q_mol)
        save_to_hdf5(driver_result, new_hdf5_file, replace=replace)

    def run(self) -> GroupedSecondQuantizedProperty:
        """
        Returns:
            GroupedSecondQuantizedProperty re-constructed from the HDF5 file.

        Raises:
            LookupError: file not found.
        """
        hdf5_file = self._get_path()

        legacy_hdf5_file = False

        with h5py.File(hdf5_file, "r") as file:
            if "origin_driver" in file.keys():
                legacy_hdf5_file = True
                LOGGER.warning(
                    "Your HDF5 file contains the legacy QMolecule object! You should consider "
                    "converting it to the new property framework. See also HDF5Driver.convert"
                )

        if legacy_hdf5_file:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            molecule = QMolecule(hdf5_file)
            warnings.filterwarnings("default", category=DeprecationWarning)
            molecule.load()
            return ElectronicStructureDriverResult.from_legacy_driver_result(molecule)

        driver_result = load_from_hdf5(hdf5_file)
        return driver_result
