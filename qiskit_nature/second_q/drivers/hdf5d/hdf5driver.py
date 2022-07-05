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
import pathlib
import warnings

import h5py

from qiskit_nature import QiskitNatureError
from qiskit_nature.deprecation import warn_deprecated, DeprecatedType
from qiskit_nature.hdf5 import load_from_hdf5, save_to_hdf5
from qiskit_nature.second_q.properties.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)
from qiskit_nature.second_q.properties import (
    ElectronicStructureDriverResult,
)

from qiskit_nature.second_q._qmolecule import QMolecule
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

    def _get_path(self) -> pathlib.Path:
        """Returns the absolute path to the HDF5 file.

        Returns:
            The absolute path to the HDF5 file.

        Raises:
            LookupError: file not found.
        """
        hdf5_file = pathlib.Path(self._hdf5_input)
        if self.work_path is not None and not hdf5_file.is_absolute():
            hdf5_file = pathlib.Path(self.work_path) / hdf5_file

        if not hdf5_file.is_file():
            raise LookupError(f"HDF5 file not found: {hdf5_file}")

        return hdf5_file

    def convert(self, replace: bool = False) -> None:
        """Converts a legacy QMolecule HDF5 file into the new Property-framework.

        Args:
            replace: if True, will replace the original HDF5 file. Otherwise `_new.hdf5` will be
                used as a suffix.

        Raises:
            LookupError: file not found.
        """
        hdf5_file = self._get_path()

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        q_mol = QMolecule(hdf5_file)
        warnings.filterwarnings("default", category=DeprecationWarning)
        q_mol.load()

        new_hdf5_file = hdf5_file
        if not replace:
            new_hdf5_file = hdf5_file.with_name(str(hdf5_file.stem) + "_new.hdf5")

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        driver_result = ElectronicStructureDriverResult.from_legacy_driver_result(q_mol)
        warnings.filterwarnings("default", category=DeprecationWarning)
        save_to_hdf5(driver_result, str(new_hdf5_file), replace=replace)

    def run(self) -> GroupedSecondQuantizedProperty:
        """
        Returns:
            GroupedSecondQuantizedProperty re-constructed from the HDF5 file.

        Raises:
            LookupError: file not found.
            QiskitNatureError: if the HDF5 file did not contain a GroupedSecondQuantizedProperty.
        """
        hdf5_file = self._get_path()

        legacy_hdf5_file = False

        with h5py.File(hdf5_file, "r") as file:
            if "origin_driver" in file.keys():
                legacy_hdf5_file = True
                warn_deprecated(
                    "0.4.0",
                    DeprecatedType.METHOD,
                    "HDF5Driver.run with legacy HDF5 file",
                    additional_msg=(
                        ". Your HDF5 file contains the legacy QMolecule object! You should consider "
                        "converting it to the new property framework. See also HDF5Driver.convert"
                    ),
                )

        if legacy_hdf5_file:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            try:
                molecule = QMolecule(hdf5_file)
                molecule.load()
                return ElectronicStructureDriverResult.from_legacy_driver_result(molecule)
            finally:
                warnings.filterwarnings("default", category=DeprecationWarning)

        driver_result = load_from_hdf5(str(hdf5_file))

        if not isinstance(driver_result, GroupedSecondQuantizedProperty):
            raise QiskitNatureError(
                f"Expected a GroupedSecondQuantizedProperty but found a {type(driver_result)} "
                "object instead."
            )

        return driver_result
