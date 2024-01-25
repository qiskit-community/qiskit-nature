# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The hexagonal lattice"""

from __future__ import annotations

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
            rows: Number of hexagons in the x direction.
            cols: Number of hexagons in the y direction.
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
        for idx in range(graph.num_edges()):
            graph.update_edge_by_index(idx, self._edge_parameter)

        # Add self loops
        for node in range(graph.num_nodes()):
            graph.add_edges_from([(node, node, self._onsite_parameter)])

        super().__init__(graph)

        self.pos = self._default_position()

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

    def _default_position(self) -> dict[int, tuple[int, int]]:
        """Return a dictionary of default positions for visualization of
            a one- or two-dimensional lattice.

        Returns:
            A dictionary where the keys are the labels of lattice points, and the values are
            two-dimensional coordinates.
        """
        pos = {}
        rowlen = 2 * self._rows + 2
        collen = self._cols + 1
        x_adjust = 0

        for i in range(collen):
            x_adjust += 1
            for j in range(rowlen):
                idx = i * rowlen + j - 1
                x = i

                # plot the y coords to form heavy hex shape
                if i == 0:
                    y = j - 1
                elif (self._cols % 2 == 0) and (i == self._cols):
                    y = j + 1
                else:
                    y = j

                # even numbered nodes in the first, last and odd numbered columns need to be
                # shifted to the right
                if i == 0 or (i == self._cols) or (i % 2 != 0):
                    if idx % 2 == 0:
                        x = i + x_adjust
                    else:
                        x = i + i
                # odd numbered nodes that aren't in the first, last or odd numbered columns
                # need to be shifted to the right
                else:
                    if idx % 2 == 0:
                        x = i + i
                    else:
                        x = i + x_adjust

                pos[idx] = (x, y)

        return pos
