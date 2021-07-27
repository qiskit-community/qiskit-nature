# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for fermionic driver modules.
"""

from abc import abstractmethod
from typing import Optional
from enum import Enum

from .qmolecule import QMolecule
from .base_driver import BaseDriver


class MethodType(Enum):
    """MethodType Enum

    The HF-style methods are common names which are likely available everywhere.
    The KS-style methods are not available for all drivers. Please check the specific driver
    documentation for details.
    """

    RHF = "rhf"
    ROHF = "rohf"
    UHF = "uhf"
    RKS = "rks"
    ROKS = "roks"
    UKS = "uks"


class FermionicDriver(BaseDriver):
    """
    Base class for Qiskit Nature's fermionic drivers.
    """

    def __init__(
        self,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        supports_molecule: bool = False,
    ) -> None:
        """
        Args:
            basis: basis set
            method: Hartree-Fock Method type
            supports_molecule: Indicates if driver supports molecule
        """
        self._method = method
        super().__init__(basis=basis, supports_molecule=supports_molecule)

    @property
    def method(self) -> MethodType:
        """return Hartree-Fock method"""
        return self._method

    @method.setter
    def method(self, value: MethodType) -> None:
        """set Hartree-Fock method"""
        self._method = value

    @abstractmethod
    def run(self) -> QMolecule:
        """
        Runs driver to produce a QMolecule output.

        Returns:
            A QMolecule containing the molecular data.
        """
        pass
