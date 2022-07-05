# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for electronic structure driver modules.
"""

from abc import abstractmethod
from enum import Enum

from qiskit_nature.second_q.properties import (
    ElectronicStructureDriverResult,
)
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


class ElectronicStructureDriver(BaseDriver):
    """
    Base class for Qiskit Nature's electronic structure drivers.
    """

    @abstractmethod
    def run(self) -> ElectronicStructureDriverResult:
        """Returns a ElectronicStructureDriverResult output as produced by the driver."""
        pass
