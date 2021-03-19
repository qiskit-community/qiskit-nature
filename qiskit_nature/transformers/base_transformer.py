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

"""The Base Operator Transformer interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTransformer(ABC):
    """The interface for implementing methods which map from one `QMolecule` or
    'WatsonHamiltonian' to another. These methods may or may not affect the size of the Hilbert
    space.
    """

    @abstractmethod
    def transform(self, molecule_data: Any):
        """Transforms one `QMolecule` or 'WatsonHamiltonian' into another one. This may or may
        not affect the size of the Hilbert space.

        Args:
            molecule_data: the `QMolecule` or 'WatsonHamiltonian' to be transformed.

        Returns:
            A new `QMolecule` or 'WatsonHamiltonian' instance.
        """
        raise NotImplementedError()
