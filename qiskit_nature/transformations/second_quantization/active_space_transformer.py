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

"""The Active-Space Reduction interface."""

from typing import Optional

from qiskit_nature.drivers.qmolecule import QMolecule
from .second_quantized_transformer import BaseTransformer


class ActiveSpaceTransformer(BaseTransformer):
    """The Active-Space reduction."""

    def __init__(self, num_electrons: int, num_orbitals: int, num_alpha: Optional[int] = None):
        """

        Args:
            num_electrons: the number of active electrons.
            num_orbitals: the number of active orbitals.
            num_alpha: the optional number of active alpha-spin electrons.
        """
        self.num_electrons = num_electrons
        self.num_orbitals = num_orbitals
        self.num_alpha = num_alpha

    def transform(self, q_molecule: QMolecule) -> QMolecule:
        """Reduces the given `QMolecule` to a given active space.

        Args:
            q_molecule: the `QMolecule` to be transformed.

        Returns:
            A new `QMolecule` instance.
        """
        # TODO
        raise NotImplementedError()
