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
import numpy as np

from qiskit_nature.second_q.operators import FermionicOp

from .lattice_model import LatticeModel
from .lattices import Lattice


class FermiHubbardModel(LatticeModel):
    """The Fermi-Hubbard model."""

    def __init__(self, lattice: Lattice, onsite_interaction: complex) -> None:
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

    @property
    def register_length(self) -> int:
        return 2 * self._lattice.num_nodes

    def second_q_op(self) -> FermionicOp:
        """Return the Hamiltonian of the Fermi-Hubbard model in terms of `FermionicOp`.

        Returns:
            FermionicOp: The Hamiltonian of the Fermi-Hubbard model.
        """
        kinetic_ham = {}
        interaction_ham = {}
        weighted_edge_list = self._lattice.weighted_edge_list
        register_length = 2 * self._lattice.num_nodes
        # kinetic terms
        for spin in range(2):
            for node_a, node_b, weight in weighted_edge_list:
                if node_a == node_b:
                    index = 2 * node_a + spin
                    kinetic_ham[f"+_{index} -_{index}"] = weight

                else:
                    if node_a < node_b:
                        index_left = 2 * node_a + spin
                        index_right = 2 * node_b + spin
                        hopping_parameter = weight
                    elif node_a > node_b:
                        index_left = 2 * node_b + spin
                        index_right = 2 * node_a + spin
                        hopping_parameter = np.conjugate(weight)
                    kinetic_ham[f"+_{index_left} -_{index_right}"] = hopping_parameter
                    kinetic_ham[f"-_{index_left} +_{index_right}"] = -np.conjugate(
                        hopping_parameter
                    )
        # on-site interaction terms
        for node in self._lattice.node_indexes:
            index_up = 2 * node
            index_down = 2 * node + 1
            interaction_ham[
                f"+_{index_up} -_{index_up} +_{index_down} -_{index_down}"
            ] = self._onsite_interaction

        ham = {**kinetic_ham, **interaction_ham}

        return FermionicOp(ham, num_spin_orbitals=register_length, copy=False)
