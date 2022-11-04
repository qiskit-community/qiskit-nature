# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The lattice model class."""

import numpy as np

from .hamiltonian import Hamiltonian
from .lattices import Lattice


class LatticeModel(Hamiltonian):
    """The lattice model."""

    def __init__(self, lattice: Lattice) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
        """
        self._lattice = lattice

    @property
    def lattice(self) -> Lattice:
        """Return the input lattice.

        Note, that this returns the internal lattice object without copying it. Thus, changing the
        returned lattice object, will affect the internal lattice stored in this Hamiltonian.
        """
        return self._lattice

    def interaction_matrix(self) -> np.ndarray:
        """Return the interaction matrix

        Returns:
            The interaction matrix.
        """
        return self._lattice.to_adjacency_matrix(weighted=True)
