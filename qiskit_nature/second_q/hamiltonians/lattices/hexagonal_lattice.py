# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The hexagonal lattice"""
from rustworkx import generators

from .lattice import Lattice


class HexagonalLattice(Lattice):
    """Hexagonal lattice."""

    def __init__(
        self,
        rows: int,
        cols: int,
        edge_parameter: complex = 1.0,
        onsite_parameter: complex = 0.0,
    ) -> None:
        """
        Args:
            rows: Length of the x direction.
            cols: Length of the y direction.
            edge_parameter: Weight on all the edges, specified as a single value.
                Defaults to 1.0.
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
        """
        self._rows = rows
        self._cols = cols
        self._edge_parameter = edge_parameter
        self._onsite_parameter = onsite_parameter

        graph = generators.hexagonal_lattice_graph(rows, cols, multigraph=False)

        # Add edge weights
        for idx in graph.edge_indices():
            graph.update_edge_by_index(idx, self._edge_parameter)

        # Add self loops
        for node in graph.node_indices():
            graph.add_edges_from([(node, node, self._onsite_parameter)])

        super().__init__(graph)

    @property
    def edge_parameter(self) -> complex:
        """Weights on all edges.

        Returns:
            the parameter for the edges.
        """
        return self._edge_parameter

    @property
    def onsite_parameter(self) -> complex:
        """Weight on the self-loops (edges connecting a node to itself).

        Returns:
            the parameter for the self-loops.
        """
        return self._onsite_parameter
