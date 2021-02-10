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

"""The Particle/Hole Transformation interface."""

from .second_quantized_transformer import BaseTransformer
from ... import QMolecule


class ParticleHoleTransformer(BaseTransformer):
    """The Particle/Hole transformation."""

    def transform(self, q_molecule: QMolecule) -> QMolecule:
        """Transforms the given `QMolecule` into the particle/hole view.

        Args:
            q_molecule: the `QMolecule` to be transformed.

        Returns:
            A new `QMolecule` instance.
        """
        # TODO
        raise NotImplementedError()
