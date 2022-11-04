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
This module implements the abstract base class for vibrational structure driver modules.
"""

from abc import abstractmethod

from qiskit_nature.second_q.problems import VibrationalBasis, VibrationalStructureProblem

from .base_driver import BaseDriver


class VibrationalStructureDriver(BaseDriver):
    """
    Base class for Qiskit Nature's vibrational structure drivers.
    """

    @abstractmethod
    def run(self, basis: VibrationalBasis) -> VibrationalStructureProblem:  # type: ignore[override]
        # pylint: disable=arguments-differ
        """Returns a VibrationalStructureProblem output as produced by the driver.

        Args:
            basis: the basis in which to return the problem.
        """
        pass
