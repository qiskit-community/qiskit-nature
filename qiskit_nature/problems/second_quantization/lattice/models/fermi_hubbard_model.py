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

"""The Fermi-Hubbard model"""
from typing import Optional

import numpy as np
from retworkx import PyGraph

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice


class FermiHubbardModel:
    """The Fermi-Hubbard model."""

    def __init__(self, lattice: Lattice, onsite_interaction: complex) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
            onsite_interaction: The strength of the on-site interaction.
        """
        self._lattice = lattice
        self.onsite_interaction = onsite_interaction

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
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_hopping: complex,
        uniform_onsite_potential: complex,
        onsite_interaction: complex,
    ) -> "FermiHubbardModel":
        """Set a uniform hopping parameter and on-site potential over the input lattice.

        Args:
            lattice: Lattice on which the model is defined.
            uniform_hopping: The hopping parameter.
            uniform_onsite_potential: The on-site potential.
            onsite_interaction: The strength of the on-site interaction.
        Returns:
            The Fermi-Hubbard model with uniform parameters.
        """
        graph = lattice.graph
        for node_a, node_b, _ in graph.weighted_edge_list():
            if node_a != node_b:
                graph.update_edge(node_a, node_b, uniform_hopping)

        for node_a in graph.node_indexes():
            if graph.has_edge(node_a, node_a):
                graph.update_edge(node_a, node_a, uniform_onsite_potential)
            else:
                graph.add_edge(node_a, node_a, uniform_onsite_potential)

        return cls(Lattice(graph), onsite_interaction)

    @classmethod
    def from_parameters(
        cls, hopping_matrix: np.ndarray, onsite_interaction: complex
    ) -> "FermiHubbardModel":
        """Return the Hamiltonian of the Fermi-Hubbard model
        from the given hopping matrix and on-site interaction.

        Args:
            hopping_matrix: A real or complex valued square matrix.
            onsite_interaction: The strength of the on-site interaction.

        Returns:
            FermiHubbardModel: The Fermi-Hubbard model generated
                from the given hopping matrix and on-site interaction.

        Raises:
            ValueError: If the hopping matrix is not square matrix,
                it is invalid.
        """
        # make a graph from the hopping matrix.
        # This should be replaced by from_adjacency_matrix of retworkx.
        shape = hopping_matrix.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            graph = PyGraph(multigraph=False)
            graph.add_nodes_from(range(shape[0]))
            for source_index in range(shape[0]):
                for target_index in range(source_index, shape[0]):
                    weight = hopping_matrix[source_index, target_index]
                    if not weight == 0.0:
                        graph.add_edge(source_index, target_index, weight)
            lattice = Lattice(graph)
            return cls(lattice, onsite_interaction)
        else:
            raise ValueError(
                f"Invalid shape of `hopping_matrix`, {shape},  is given."
                "It must be a square matrix."
            )

    def second_q_ops(self, display_format: Optional[str] = None) -> FermionicOp:
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
            interaction_ham.append((f"N_{index_up} N_{index_down}", self.onsite_interaction))

        ham = kinetic_ham + interaction_ham

        return FermionicOp(ham, register_length=register_length, display_format=display_format)
