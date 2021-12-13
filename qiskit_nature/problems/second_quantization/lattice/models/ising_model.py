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

"""The Ising model"""
from typing import Optional

import numpy as np

from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice

from .lattice_model import LatticeModel


class IsingModel(LatticeModel):
    """The Ising model."""

    @classmethod
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_hopping: complex,
        uniform_onsite_potential: complex,
        onsite_interaction: complex = None,
    ) -> "IsingModel":
        """Set a uniform hopping parameter and on-site potential over the input lattice.

        Args:
            lattice: Lattice on which the model is defined.
            uniform_hopping: The hopping parameter. (or uniform_interaction)
            uniform_onsite_potential: The on-site potential. (or uniform_external_field)
            onsite_interaction: The strength of the on-site interaction, the Ising model does not
                have an on-site interaction.

        Returns:
            The Ising model with uniform parameters.

        Raises:
            Warning: If the on-site interaction is not None.
        """
        if onsite_interaction is not None:
            raise Warning(
                "The Ising model does not have on-site interactions. Provided onsite-interaction "
                "parameter will be ignored."
            )
        return super().uniform_parameters(
            lattice, uniform_hopping, uniform_onsite_potential, None
        )

    @classmethod
    def from_parameters(
        cls, coupling_matrix: np.ndarray, onsite_interaction: complex = None
    ) -> "IsingModel":
        """Return the Hamiltonian of the Ising model from the given coupling matrix.

        Args:
            coupling_matrix: A real or complex valued square symmetric matrix.
            onsite_interaction: The strength of the on-site interaction, the Ising model does not
                have an on-site interaction.

        Returns:
            IsingModel: The Ising model generated from the given coupling matrix.

        Raises:
            Warning: If the on-site interaction is not None.
            ValueError: If the coupling matrix is not square matrix, it is invalid.
        """
        if onsite_interaction is not None:
            raise Warning(
                "The Ising model does not have on-site interactions. Provided onsite-interaction "
                "parameter will be ignored."
            )
        return super().from_parameters(coupling_matrix, None)

    def second_q_ops(self, display_format: Optional[str] = None) -> SpinOp:
        """Return the Hamiltonian of the Ising model in terms of `SpinOp`.

        Args:
            display_format: Not supported for Spin operators.

        Returns:
            SpinOp: The Hamiltonian of the Ising model.

        Raises:
            Warning: If display-format is not None.
        """
        if display_format is not None:
            raise Warning(
                "Spin operators do not support display-format. Provided display-format "
                "parameter will be ignored."
            )
        ham = []
        weighted_edge_list = self._lattice.weighted_edge_list
        register_length = self._lattice.num_nodes
        # kinetic terms
        for node_a, node_b, weight in weighted_edge_list:
            if node_a == node_b:
                index = node_a
                ham.append((f"X_{index}", weight))

            else:
                index_left = node_a
                index_right = node_b
                coupling_parameter = weight
                ham.append((f"Z_{index_left} Z_{index_right}", coupling_parameter))

        return SpinOp(ham, spin=1 / 2, register_length=register_length)
