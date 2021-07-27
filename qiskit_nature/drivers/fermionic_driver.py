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

from .qmolecule import QMolecule
from .base_driver import BaseDriver
from ..deprecation import (
    DeprecatedEnum,
    DeprecatedEnumMeta,
    DeprecatedType,
    warn_deprecated_same_type_name,
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
    def run(self) -> QMolecule:
        """
        Runs driver to produce a QMolecule output.

        Returns:
            A QMolecule containing the molecular data.
        """
        pass
