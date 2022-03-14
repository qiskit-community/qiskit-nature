# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The initial point generator interface."""

from abc import ABC, abstractmethod

import numpy as np


class PointGenerator(ABC):
    """The initial point generator interface."""

    @property
    @abstractmethod
    def initial_point(self) -> np.ndarray:
        """Returns the initial point."""
        raise NotImplementedError
