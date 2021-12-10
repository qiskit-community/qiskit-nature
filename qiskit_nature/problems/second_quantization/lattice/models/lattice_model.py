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

"""The Lattice Model class."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice


class LatticeModel(ABC):
    """Lattice Model"""

    def __init__(self, lattice: Lattice) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
        """
        self._lattice = lattice

    @property
    def lattice(self) -> Lattice:
        """Return a copy of the input lattice."""
        return self._lattice.copy()

    def hopping_matrix(self) -> np.ndarray:
        """Return the hopping matrix
        Returns:
            The hopping matrix.
        """
        return self._lattice.to_adjacency_matrix(weighted=True)

    @classmethod
    @abstractmethod
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_hopping: complex,
        uniform_onsite_potential: complex,
        onsite_interaction: complex,
    ) -> "LatticeModel":
        """Set a uniform hopping parameter and on-site potential over the input lattice.

        Args:
            lattice: Lattice on which the model is defined.
            uniform_hopping: The hopping parameter.
            uniform_onsite_potential: The on-site potential.
            onsite_interaction: The strength of the on-site interaction.
        Returns:
            The Lattice model with uniform parameters.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_parameters(
        cls, hopping_matrix: np.ndarray, onsite_interaction: complex
    ) -> "LatticeModel":
        """Return the Hamiltonian of the Lattice model
        from the given hopping matrix and on-site interaction.

        Args:
            hopping_matrix: A real or complex valued square matrix.
            onsite_interaction: The strength of the on-site interaction.

        Returns:
            LatticeModel: The Lattice model generated from the given hopping
                matrix and on-site interaction.

        Raises:
            ValueError: If the hopping matrix is not square matrix,
                it is invalid.
        """
        raise NotImplementedError()

    @abstractmethod
    def second_q_ops(self, display_format: Optional[str] = None) -> SecondQuantizedOp:
        """Return the Hamiltonian of the Lattice model in terms of `SecondQuantizedOp`.

        Args:
            display_format: If sparse, the label is represented sparsely during output.
                If dense, the label is represented densely during output. Defaults to "dense".

        Returns:
            SecondQuantizedOp: The Hamiltonian of the Lattice model.
        """
        raise NotImplementedError()
