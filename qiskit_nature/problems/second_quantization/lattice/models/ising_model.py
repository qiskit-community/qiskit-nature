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
import numpy as np
from retworkx import PyGraph

from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice


class IsingModel:
    """The Ising model."""

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
    def uniform_parameters(
        cls,
        lattice: Lattice,
        uniform_hopping: complex,
        uniform_onsite_potential: complex,
    ) -> "IsingModel":
        """Set a uniform hopping parameter and on-site potential over the input lattice.
        Args:
            lattice: Lattice on which the model is defined.
            uniform_hopping: The hopping parameter. (or uniform_interaction)
            uniform_onsite_potential: The on-site potential. (or uniform_external_field)
        Returns:
            The Ising model with uniform parameters.
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

        return cls(Lattice(graph))

    @classmethod
    def from_parameters(cls, hopping_matrix: np.ndarray) -> "IsingModel":
        """Return the Hamiltonian of the Ising model from the given hopping matrix.
        Args:
            hopping_matrix: A real or complex valued square symmetric matrix.
        Returns:
            IsingModel: The Ising model generated from the given hopping matrix.
        Raises:
            ValueError: If the hopping matrix is not square matrix, it is invalid.
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
            return cls(lattice)
        else:
            raise ValueError(
                f"Invalid shape of `hopping_matrix`, {shape},  is given."
                "It must be a square matrix."
            )

    def second_q_ops(self) -> SpinOp:
        """Return the Hamiltonian of the Ising model in terms of `SpinOp`.
        Returns:
            SpinOp: The Hamiltonian of the Ising model.
        """
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
                hopping_parameter = weight
                ham.append((f"Z_{index_left} Z_{index_right}", hopping_parameter))

        return SpinOp(ham, spin=1, register_length=register_length)
