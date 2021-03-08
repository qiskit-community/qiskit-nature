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
"""The Base Problem class."""
from abc import ABC, abstractmethod
from typing import List, Optional

from qiskit_nature.drivers import BaseDriver
from qiskit_nature.transformers import BaseTransformer


class BaseProblem(ABC):
    """Base Problem"""

    def __init__(self, driver: BaseDriver,
                 transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            driver: A driver encoding the molecule information.
            transformers: A list of transformations to be applied to the molecule.
        """
        if transformers is None:
            transformers = []
        self.driver = driver
        self.transformers = transformers

    @abstractmethod
    def second_q_ops(self):
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: ... .
        """
        return

    @abstractmethod
    def _transform(self, molecule):
        return
