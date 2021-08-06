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

"""
This module implements the abstract base class for driver modules.
"""

from abc import ABC, abstractmethod

from qiskit_nature.properties.second_quantization import GroupedSecondQuantizedProperty


class BaseDriver(ABC):
    """
    Base class for Qiskit Nature drivers.
    """

    @abstractmethod
    def run(self) -> GroupedSecondQuantizedProperty:
        """
        Runs a driver to produce an output data structure.
        """
        raise NotImplementedError()
