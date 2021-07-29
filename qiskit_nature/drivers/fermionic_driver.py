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

from .qmolecule import QMolecule
from .base_driver import BaseDriver
from .molecule import Molecule
from ..deprecation import (
    DeprecatedEnum,
    DeprecatedEnumMeta,
    DeprecatedType,
    warn_deprecated_same_type_name,
    warn_deprecated,
)


class HFMethodType(DeprecatedEnum, metaclass=DeprecatedEnumMeta):
    """HFMethodType Enum"""

    RHF = "rhf"
    ROHF = "rohf"
    UHF = "uhf"

    def deprecate(self):
        """show deprecate message"""
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.ENUM,
            self.__class__.__name__,
            "from qiskit_nature.drivers.second_quantization",
            3,
        )


class FermionicDriver(BaseDriver):
    """
    Base class for Qiskit Nature's fermionic drivers.
    """

    @abstractmethod
    def __init__(
        self,
        molecule: Optional[Molecule] = None,
        basis: str = "sto3g",
        hf_method: str = "rhf",
        supports_molecule: bool = False,
    ) -> None:
        """
        Args:
            molecule: molecule
            basis: basis set
            hf_method: Hartree-Fock Method type
            supports_molecule: Indicates if driver supports molecule
        """
        super().__init__(
            molecule, basis=basis, hf_method=hf_method, supports_molecule=supports_molecule
        )
        warn_deprecated(
            "0.2.0",
            old_type=DeprecatedType.CLASS,
            old_name="FermionicDriver",
            new_name="ElectronicStructureDriver",
            additional_msg="from qiskit_nature.drivers.second_quantization",
        )

    @abstractmethod
    def run(self) -> QMolecule:
        """
        Runs driver to produce a QMolecule output.

        Returns:
            A QMolecule containing the molecular data.
        """
        pass
