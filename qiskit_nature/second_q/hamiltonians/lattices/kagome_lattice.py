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

from __future__ import annotations

from dataclasses import asdict
from itertools import product

import numpy as np
from rustworkx import PyGraph

from .boundary_condition import BoundaryCondition
from .lattice import Lattice, LatticeDrawStyle


class KagomeLattice(Lattice):
    r"""The two-dimensional kagome lattice.

    The kagome lattice is a two-dimensional Bravais lattice formed by tiling together
    equilateral triangles and regular hexagons in an alternating pattern. The lattice is
    spanned by the primitive lattice vectors :math:`\vec{a}_{1} = (1, 0)^{\top}` and
    :math:`\vec{a}_{2} = (1/2, \sqrt{3}/2)^{\top}` with each unit cell consisting of three
    lattice sites located at :math:`\vec{r}_0 = \mathbf{0}`, :math:`\vec{r}_1 = 2\vec{a}_{1}`
    and :math:`\vec{r}_2 = 2 \vec{a}_{2}`, respectively.

    This class allows for the simple construction of kagome lattices. For example,

    .. code-block:: python

        from qiskit_nature.second_q.hamiltonians.lattices import (
            BoundaryCondition,
            KagomeLattice,
        )

        kagome = KagomeLattice(
            5,
            4,
            edge_parameter = 1.0,
            onsite_parameter = 2.0,
            boundary_condition = BoundaryCondition.PERIODIC
        )

    instantiates a kagome lattice with 5 and 4 unit cells in the x and y direction,
    respectively, which has weights 1.0 on all edges and weights 2.0 on self-loops.
    The boundary conditions are periodic for the entire lattice.

     References:
        - `Kagome Lattice @ wikipedia <https://en.wikipedia.org/wiki/Trihexagonal_tiling>`_
        - `Bravais Lattice @ wikipedia <https://en.wikipedia.org/wiki/Bravais_lattice>`_
    """

    # Dimension of lattice
    _dim = 2

    # Number of sites in a unit cell
    _num_sites_per_cell = 3

    # Relative positions (relative to site 0) of sites in a unit cell
    _cell_positions = np.array([[0, 0], [1, 0], [1 / 2, np.sqrt(3) / 2]])

    # Primitive translation vectors in each direction
    _basis = np.array([[2, 0], [1, np.sqrt(3)]])

    def _coordinate_to_index(self, coord: np.ndarray) -> int:
        """Convert the coordinate of a lattice point to an integer for labeling.

            When self.size=(l0, l1), then a coordinate (x0, x1) is converted as
            x0 + x1*l0.

        Args:
            coord: Input coordinate to be converted.

        Returns:
            int: Return x0 + x1*l0 when coord=np.array([x0, x1]) and self.size=(l0, l1).
        """
        dim = self._dim
        size = self._size
        base = np.array([np.prod(size[:i]) for i in range(dim)], dtype=int)
        return np.dot(coord, base).item()

    def _self_loops(self) -> list[tuple[int, int, complex]]:
        """Return a list consisting of the self-loops on all the nodes.

        Returns:
            list[tuple[int, int, complex]] : list of the self-loops.
        """
        size = self._size
        onsite_parameter = self._onsite_parameter
        num_nodes = self._num_sites_per_cell * np.prod(size)
        return [(node_a, node_a, onsite_parameter) for node_a in range(num_nodes)]

    def _bulk_edges(self) -> list[tuple[int, int, complex]]:
        """Return a list consisting of the edges in the bulk, which don't cross the boundaries.

        Returns:
            list[tuple[int, int, complex]] : list of weighted edges that don't cross the boundaries.
        """
        size = self._size
        edge_parameter = self._edge_parameter
        num_sites_per_cell = self._num_sites_per_cell
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

    def _boundary_edges(self) -> list[tuple[int, int, complex]]:
        """Return a list consisting of the edges that cross the boundaries
            depending on the boundary conditions.

        Raises:
            ValueError: Given boundary condition is invalid values.

        Returns:
            list[tuple[int, int, complex]]: list of weighted edges that cross the boundaries.
        """
        list_of_edges = []
        size = self._size
        edge_parameter = self._edge_parameter
        num_sites_per_cell = self._num_sites_per_cell
        boundary_condition = self._boundary_condition
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

    def _default_position(self, with_boundaries: bool = True) -> dict[int, list[float]]:
        """Return a dictionary of default positions for visualization of a two-dimensional lattice.

        Returns:
            dict[int, list[float]] : The keys are the labels of lattice points,
                and the values are two-dimensional coordinates.
        """
        size = self._size
        boundary_condition = self._boundary_condition
        num_sites_per_cell = self._num_sites_per_cell
        pos = {}
        width = 0.0
        if with_boundaries and boundary_condition == BoundaryCondition.PERIODIC:
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

            for i in range(num_sites_per_cell):
                node_i = num_sites_per_cell * cell_idx + i
                pos[node_i] = (np.dot(cell_coord, self._basis) + self._cell_positions[i]).tolist()
        return pos

    def _style_pos(self) -> dict[int, list[float]]:
        """Return a dictionary of positions for visualization of a two-dimensional lattice without
            boundaries.

        Returns:
            dict[int, list[float]] : The keys are the labels of lattice points,
                and the values are two-dimensional coordinates.
        """
        size = self._size
        num_sites_per_cell = self._num_sites_per_cell
        basis = self._basis
        cell_positions = self._cell_positions
        pos = {}

        for cell_idx in range(np.prod(size)):
            # maps an cell index to two-dimensional coordinate
            # the positions are shifted so that the edges between boundaries can be seen
            # for the periodic cases.
            cell_coord = np.array(divmod(cell_idx, size[0])[::-1])

            for i in range(num_sites_per_cell):
                node_i = num_sites_per_cell * cell_idx + i
                pos[node_i] = (np.dot(cell_coord, basis) + cell_positions[i]).tolist()
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
            rows: Number of unit cells in the x direction.
            cols: Number of unit cells in the y direction.
            edge_parameter: Weight on all the edges, specified as a single value
                Defaults to 1.0,
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
            boundary_condition: Boundary condition for the lattice.
                The available boundary conditions are:
                ``BoundaryCondition.OPEN``, ``BoundaryCondition.PERIODIC``.
                Defaults to ``BoundaryCondition.OPEN``.
        """
        self._rows = rows
        self._cols = cols
        self._size = (rows, cols)
        self._boundary_condition = boundary_condition
        self._edge_parameter = edge_parameter
        self._onsite_parameter = onsite_parameter

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(self._num_sites_per_cell * np.prod(self._size)))

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

    @property
    def rows(self) -> int:
        """Number of unit cells in the x direction.

        Returns:
            the number.
        """
        return self._rows

    @property
    def cols(self) -> int:
        """Number of unit cells in the y direction.

        Returns:
            the number
        """
        return self._cols

    @property
    def size(self) -> tuple[int, int]:
        """Number of unit cells in the x and y direction, respectively.

        Returns:
            the size.
        """
        return self._size

    @property
    def edge_parameter(self) -> complex:
        """Weights on all edges.

        Returns:
            the parameter for the edges.
        """
        return self._edge_parameter

    @property
    def onsite_parameter(self) -> complex:
        """Weight on the self-loops.

        Returns:
            the parameter for the self-loops.
        """
        return self._onsite_parameter

    @property
    def boundary_condition(self) -> BoundaryCondition:
        """Boundary condition for the entire lattice.

        Returns:
            the boundary condition.
        """
        return self._boundary_condition

    def draw_without_boundary(
        self,
        *,
        self_loop: bool = False,
        style: LatticeDrawStyle | None = None,
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

        if style is None:
            style = LatticeDrawStyle()
        elif not isinstance(style, LatticeDrawStyle):
            style = LatticeDrawStyle(**style)

        if style.pos is None:
            style.pos = self._default_position(with_boundaries=False)

        graph.remove_edges_from(self.boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
