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

"""The Seniority-Zero Transformer interface."""

from .base_transformer import BaseTransformer
from ..drivers import QMolecule


class SeniorityZeroTransformer(BaseTransformer):
    """The Seniority-Zero transformer."""

    def transform(self, q_molecule: QMolecule) -> QMolecule:
        """Transforms the given `QMolecule` into a seniority-zero (i.e.
        restricted-spin) variant.

        Args:
            q_molecule: the `QMolecule` to be transformed.

        Returns:
            A new `QMolecule` instance.
        """
        # TODO
        raise NotImplementedError()
