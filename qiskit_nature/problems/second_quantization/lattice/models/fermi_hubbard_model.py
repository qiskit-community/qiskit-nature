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

"""The Fermi-Hubbard model"""
from typing import Optional

import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice

from .lattice_model import LatticeModel


class FermiHubbardModel(LatticeModel):
    """The Fermi-Hubbard model."""

    def __init__(self, lattice: Lattice, onsite_interaction: complex):
        """
        Args:
            lattice: Lattice on which the model is defined.
            onsite_interaction: The strength of the on-site interaction.
        """
        super().__init__(lattice)
        self._onsite_interaction = onsite_interaction

    def hopping_matrix(self) -> np.ndarray:
        """Return the hopping matrix."""
        return self.interaction_matrix()

    @classmethod
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_interaction: complex,
        uniform_onsite_potential: complex,
        onsite_interaction: complex,
    ) -> "FermiHubbardModel":
        """Set a uniform hopping parameter and on-site potential over the input lattice.
        Args:
            lattice: Lattice on which the model is defined.
            uniform_interaction: The interaction parameter.
            uniform_onsite_potential: The on-site potential.
            onsite_interaction: The strength of the on-site interaction.
        Returns:
            The Fermi-Hubbard model with uniform parameters.
        """
        return cls(
            cls._generate_lattice_from_uniform_parameters(
                lattice, uniform_interaction, uniform_onsite_potential
            ),
            onsite_interaction,
        )

    @classmethod
    def from_parameters(
        cls, interaction_matrix: np.ndarray, onsite_interaction: complex
    ) -> "FermiHubbardModel":
        """Return the Hamiltonian of the Fermi-Hubbard model
        from the given hopping matrix and on-site interaction.
        Args:
            interaction_matrix: A real or complex valued square matrix.
            onsite_interaction: The strength of the on-site interaction.
        Returns:
            FermiHubbardModel: The Fermi-Hubbard model generated
                from the given hopping matrix and on-site interaction.
        Raises:
            ValueError: If the hopping matrix is not square matrix, it is invalid.
        """
        return cls(cls._generate_lattice_from_parameters(interaction_matrix), onsite_interaction)

    def second_q_ops(self, display_format: str = "sparse") -> FermionicOp:
        """Return the Hamiltonian of the Fermi-Hubbard model in terms of `FermionicOp`.

        Args:
            display_format: If sparse, the label is represented sparsely during output.
                If dense, the label is represented densely during output. Defaults to "dense".

        Returns:
            FermionicOp: The Hamiltonian of the Fermi-Hubbard model.
        """
        kinetic_ham = []
        interaction_ham = []
        weighted_edge_list = self._lattice.weighted_edge_list
        register_length = 2 * self._lattice.num_nodes
        # kinetic terms
        for spin in range(2):
            for node_a, node_b, weight in weighted_edge_list:
                if node_a == node_b:
                    index = 2 * node_a + spin
                    kinetic_ham.append((f"N_{index}", weight))

                else:
                    if node_a < node_b:
                        index_left = 2 * node_a + spin
                        index_right = 2 * node_b + spin
                        hopping_parameter = weight
                    elif node_a > node_b:
                        index_left = 2 * node_b + spin
                        index_right = 2 * node_a + spin
                        hopping_parameter = np.conjugate(weight)
                    kinetic_ham.append((f"+_{index_left} -_{index_right}", hopping_parameter))
                    kinetic_ham.append(
                        (f"-_{index_left} +_{index_right}", -np.conjugate(hopping_parameter))
                    )
        # on-site interaction terms
        for node in self._lattice.node_indexes:
            index_up = 2 * node
            index_down = 2 * node + 1
            interaction_ham.append((f"N_{index_up} N_{index_down}", self._onsite_interaction))

        ham = kinetic_ham + interaction_ham

        return FermionicOp(ham, register_length=register_length, display_format=display_format)
