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

"""The kagome lattice"""
from dataclasses import asdict
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from rustworkx import PyGraph

from .lattice import LatticeDrawStyle, Lattice
from .boundary_condition import BoundaryCondition


class KagomeLattice(Lattice):
    """Kagome lattice"""

    def _coordinate_to_index(self, coord: np.ndarray) -> int:
        """Convert the coordinate of a lattice point to an integer for labeling.
            When self.size=(l0, l1), then a coordinate (x0, x1) is converted as
            x0 + x1*l0.
        Args:
            coord: Input coordinate to be converted.

        Returns:
            int: Return x0 + x1*l0 when coord=np.array([x0, x1]) and self.size=(l0, l1).
        """
        dim = self.dim
        size = self.size
        base = np.array([np.prod(size[:i]) for i in range(dim)], dtype=int)
        return np.dot(coord, base).item()

    def _self_loops(self) -> List[Tuple[int, int, complex]]:
        """Return a list consisting of the self-loops on all the nodes.
        Returns:
            List[Tuple[int, int, complex]] : List of the self-loops.
        """
        size = self.size
        onsite_parameter = self.onsite_parameter
        num_nodes = self.num_sites_per_cell * np.prod(size)
        return [(node_a, node_a, onsite_parameter) for node_a in range(num_nodes)]

    def _bulk_edges(self) -> List[Tuple[int, int, complex]]:
        """Return a list consisting of the edges in the bulk, which don't cross the boundaries.

        Returns:
            List[Tuple[int, int, complex]] : List of weighted edges that don't cross the boundaries.
        """
        size = self.size
        edge_parameter = self.edge_parameter
        num_sites_per_cell = self.num_sites_per_cell
        list_of_edges = []
        rows, cols = size
        unit_cell_coordinates = list(product(*map(range, size)))

        for x, y in unit_cell_coordinates:
            # each cell is indexed by its leftmost lattice site
            cell_a_idx = self._coordinate_to_index(np.array([x, y]))

            # indices of sites within unit cell
            cell_a_0 = num_sites_per_cell * cell_a_idx
            cell_a_1 = num_sites_per_cell * cell_a_idx + 1
            cell_a_2 = num_sites_per_cell * cell_a_idx + 2

            # connect sites within a unit cell
            list_of_edges.append((cell_a_0, cell_a_1, edge_parameter))
            list_of_edges.append((cell_a_1, cell_a_2, edge_parameter))
            list_of_edges.append((cell_a_2, cell_a_0, edge_parameter))

            # one cell east if not at the east boundary
            if x != rows - 1:
                cell_b_idx = self._coordinate_to_index(np.array([x, y]) + np.array([1, 0]))
                cell_b_0 = num_sites_per_cell * cell_b_idx
                list_of_edges.append((cell_a_1, cell_b_0, edge_parameter))

            # one cell north f not at the north boundary
            if y != cols - 1:
                cell_b_idx = self._coordinate_to_index(np.array([x, y]) + np.array([0, 1]))
                cell_b_0 = num_sites_per_cell * cell_b_idx
                list_of_edges.append((cell_a_2, cell_b_0, edge_parameter))

            # one cell west and north if not at west north boundary
            if x != 0 and y != cols - 1:
                cell_b_idx = self._coordinate_to_index(np.array([x, y]) + np.array([-1, 1]))
                cell_b_1 = num_sites_per_cell * cell_b_idx + 1
                list_of_edges.append((cell_a_2, cell_b_1, edge_parameter))

        return list_of_edges

    def _boundary_edges(self) -> List[Tuple[int, int, complex]]:
        """Return a list consisting of the edges that cross the boundaries
            depending on the boundary conditions.

        Raises:
            ValueError: Given boundary condition is invalid values.
        Returns:
            List[Tuple[int, int, complex]]: List of weighted edges that cross the boundaries.
        """
        list_of_edges = []
        size = self.size
        edge_parameter = self.edge_parameter
        num_sites_per_cell = self.num_sites_per_cell
        boundary_condition = self.boundary_condition
        rows, cols = size
        # add edges when the boundary condition is periodic.
        if boundary_condition == BoundaryCondition.PERIODIC:
            # The periodic boundary condition in the x direction.
            # It makes sense only when rows is greater than 1.
            if rows > 1:
                for y in range(cols):
                    cell_a_idx = self._coordinate_to_index(np.array([rows - 1, y]))
                    cell_a_1 = num_sites_per_cell * cell_a_idx + 1

                    cell_b_idx = self._coordinate_to_index(np.array([0, y]))
                    cell_b_0 = num_sites_per_cell * cell_b_idx

                    list_of_edges.append((cell_a_1, cell_b_0, edge_parameter.conjugate()))
            # The periodic boundary condition in the y direction.
            # It makes sense only when cols is greater than 1.
            if cols > 1:
                for x in range(rows):
                    cell_a_idx = self._coordinate_to_index(np.array([x, cols - 1]))
                    cell_a_2 = num_sites_per_cell * cell_a_idx + 2

                    cell_b_idx = self._coordinate_to_index(np.array([x, 0]))
                    cell_b_0 = num_sites_per_cell * cell_b_idx

                    list_of_edges.append((cell_a_2, cell_b_0, edge_parameter.conjugate()))

            # The periodic boundary condition in the diagonal directions.
            for x in range(1, rows):
                cell_a_idx = self._coordinate_to_index(np.array([x, cols - 1]))
                cell_a_2 = num_sites_per_cell * cell_a_idx + 2

                cell_b_idx = self._coordinate_to_index(np.array([(x - 1) % rows, 0]))
                cell_b_1 = num_sites_per_cell * cell_b_idx + 1

                list_of_edges.append((cell_a_2, cell_b_1, edge_parameter.conjugate()))

            for y in range(cols - 1):
                cell_a_idx = self._coordinate_to_index(np.array([0, y]))
                cell_a_2 = num_sites_per_cell * cell_a_idx + 2

                cell_b_idx = self._coordinate_to_index(np.array([rows - 1, (y + 1) % cols]))
                cell_b_1 = num_sites_per_cell * cell_b_idx + 1

                list_of_edges.append((cell_a_2, cell_b_1, edge_parameter.conjugate()))

            # isolating x = 0, y = cols - 1 to prevent duplicating edges
            cell_a_idx = self._coordinate_to_index(np.array([0, cols - 1]))
            cell_a_2 = num_sites_per_cell * cell_a_idx + 2

            cell_b_idx = self._coordinate_to_index(np.array([rows - 1, 0]))
            cell_b_1 = num_sites_per_cell * cell_b_idx + 1

            list_of_edges.append((cell_a_2, cell_b_1, edge_parameter.conjugate()))

        elif boundary_condition == BoundaryCondition.OPEN:
            pass
        else:
            raise ValueError(
                f"Invalid `boundary condition` {boundary_condition} is given."
                "`boundary condition` must be " + " or ".join(str(bc) for bc in BoundaryCondition)
            )
        return list_of_edges

    def _default_position(self) -> Dict[int, List[float]]:
        """Return a dictionary of default positions for visualization of a two-dimensional lattice.

        Returns:
            Dict[int, List[float]] : The keys are the labels of lattice points,
                and the values are two-dimensional coordinates.
        """
        size = self.size
        boundary_condition = self.boundary_condition
        num_sites_per_cell = self.num_sites_per_cell
        pos = {}
        width = 0.0
        if boundary_condition == BoundaryCondition.PERIODIC:
            # the positions are shifted along the x- and y-direction
            # when the boundary condition is periodic.
            # The width of the shift is fixed to 0.2.
            width = 0.2
        for cell_idx in range(np.prod(size)):
            # maps an cell index to two-dimensional coordinate
            # the positions are shifted so that the edges between boundaries can be seen
            # for the periodic cases.
            cell_coord = np.array(divmod(cell_idx, size[0])[::-1]) + width * np.cos(
                np.pi * (np.array(divmod(cell_idx, size[0]))) / (np.array(size)[::-1] - 1)
            )

            for i in range(3):
                node_i = num_sites_per_cell * cell_idx + i
                pos[node_i] = (np.dot(cell_coord, self.basis) + self.cell_positions[i]).tolist()
        return pos

    def __init__(
        self,
        rows: int,
        cols: int,
        edge_parameter: complex = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.OPEN,
    ) -> None:
        """
        Args:
            rows: Length of the x direction.
            cols: Length of the y direction.
            edge_parameter: Weight on all the edges, specified as a single value
                Defaults to 1.0,
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
            boundary_condition: Boundary condition for the lattice.
                The available boundary conditions are:
                ``BoundaryCondition.OPEN``, ``BoundaryCondition.PERIODIC``.
                Defaults to ``BoundaryCondition.OPEN``.

        Raises:
            ValueError: Given size, edge parameter or boundary condition are invalid values.
        """
        self.rows = rows
        self.cols = cols
        self.size = (rows, cols)
        self.dim = 2
        self.boundary_condition = boundary_condition
        self.num_sites_per_cell = 3
        self.cell_positions = np.array([[0, 0], [1, 0], [1 / 2, np.sqrt(3) / 2]])
        self.basis = np.array([[2, 0], [1, np.sqrt(3)]])

        self.edge_parameter = edge_parameter
        self.onsite_parameter = onsite_parameter

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(self.num_sites_per_cell * np.prod(self.size)))

        # add edges excluding the boundary edges
        bulk_edges = self._bulk_edges()
        graph.add_edges_from(bulk_edges)

        # add self-loops
        self_loop_list = self._self_loops()
        graph.add_edges_from(self_loop_list)

        # add edges that cross the boundaries
        boundary_edge_list = self._boundary_edges()
        graph.add_edges_from(boundary_edge_list)

        # a list of edges that depend on the boundary condition
        self.boundary_edges = [(edge[0], edge[1]) for edge in boundary_edge_list]
        super().__init__(graph)
        # default position
        self.pos = self._default_position()

    def draw_without_boundary(
        self,
        *,
        self_loop: bool = False,
        style: Optional[LatticeDrawStyle] = None,
    ):
        r"""Draw the lattice with no edges between the boundaries.

        Args:
            self_loop: Draw self-loops in the lattice. Defaults to False.
            style : Styles for rustworkx.visualization.mpl_draw.
                Please see
                https://qiskit.org/documentation/rustworkx/stubs/rustworkx.visualization.mpl_draw.html#rustworkx.visualization.mpl_draw
                for details.
        """
        graph = self.graph
        num_sites_per_cell = self.num_sites_per_cell
        size = self.size

        if style is None:
            style = LatticeDrawStyle()
        elif not isinstance(style, LatticeDrawStyle):
            style = LatticeDrawStyle(**style)

        if style.pos is None:
            style.pos = {}
            for cell_idx in range(np.prod(size)):
                cell_coord = np.array(divmod(cell_idx, size[0])[::-1])

                for i in range(3):
                    node_i = num_sites_per_cell * cell_idx + i
                    style.pos[node_i] = (
                        np.dot(cell_coord, self.basis) + self.cell_positions[i]
                    ).tolist()

        graph.remove_edges_from(self.boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
