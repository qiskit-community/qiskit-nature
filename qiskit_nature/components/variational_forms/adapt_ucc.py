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
"""
TODO.
"""

import logging

from .adaptive_ansatz import AdaptiveAnsatz
from .ucc import UCC

logger = logging.getLogger(__name__)


class AdaptUCC(UCC, AdaptiveAnsatz):
    """The Adaptive Unitary Coupled-Cluster Ansatz.

    To be used in combination with an adaptive algorithm like e.g. :code:`AdaptVQE`.
    """

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        pass

    def _build(self) -> None:
        pass
