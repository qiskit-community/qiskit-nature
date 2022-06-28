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
from retworkx import PyGraph

from qiskit_nature.second_quantization.operators import SecondQuantizedOp
from qiskit_nature.second_quantization.problems.lattice.lattices import Lattice


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

    def interaction_matrix(self) -> np.ndarray:
        """Return the interaction matrix

        Returns:
            The interaction matrix.
        """
        return self._lattice.to_adjacency_matrix(weighted=True)

    @staticmethod
    def _generate_lattice_from_uniform_parameters(
        lattice: Lattice,
        uniform_interaction: complex,
        uniform_onsite_potential: complex,
    ) -> Lattice:
        graph = lattice.graph
        for node_a, node_b, _ in graph.weighted_edge_list():
            if node_a != node_b:
                graph.update_edge(node_a, node_b, uniform_interaction)

        for node_a in graph.node_indexes():
            if graph.has_edge(node_a, node_a):
                graph.update_edge(node_a, node_a, uniform_onsite_potential)
            else:
                graph.add_edge(node_a, node_a, uniform_onsite_potential)

        return Lattice(graph)

    @staticmethod
    def _generate_lattice_from_parameters(interaction_matrix: np.ndarray):
        # make a graph from the interaction matrix.
        # This should be replaced by from_adjacency_matrix of retworkx.
        shape = interaction_matrix.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"Invalid shape of `interaction_matrix`, {shape},  is given."
                "It must be a square matrix."
            )

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(shape[0]))
        for source_index in range(shape[0]):
            for target_index in range(source_index, shape[0]):
                weight = interaction_matrix[source_index, target_index]
                if not weight == 0.0:
                    graph.add_edge(source_index, target_index, weight)
        return Lattice(graph)

    @abstractmethod
    def second_q_ops(self, display_format: Optional[str] = None) -> SecondQuantizedOp:
        """Return the Hamiltonian of the Lattice model in terms of `SecondQuantizedOp`.

        Args:
            display_format: If sparse, the label is represented sparsely during output.
                If dense, the label is represented densely during output. Defaults to "dense".

        Returns:
            SecondQuantizedOp: The Hamiltonian of the Lattice model.
        """
