# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
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

    def __init__(
        self,
        rows: int,
        cols: int,
        edge_parameter: complex = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition
        | tuple[BoundaryCondition, BoundaryCondition] = BoundaryCondition.OPEN,
    ) -> None:
        """
        Args:
            rows: Number of unit cells in the x direction.
            cols: Number of unit cells in the y direction.
            edge_parameter: Weight on all the edges, specified as a single value
                Defaults to 1.0,
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
            boundary_condition: Boundary condition for each direction.
                The available boundary conditions are:
                :attr:`.BoundaryCondition.OPEN`, :attr:`.BoundaryCondition.PERIODIC`.
                Defaults to :attr:`.BoundaryCondition.OPEN`.

        Raises:
            ValueError: When edge parameter or boundary condition is a tuple,
                the length of that is not the same as that of size.
        """
        self._rows = rows
        self._cols = cols
        self._size = (rows, cols)
        self._edge_parameter = edge_parameter
        self._onsite_parameter = onsite_parameter

        if isinstance(boundary_condition, BoundaryCondition):
            boundary_condition = (boundary_condition, boundary_condition)
        elif isinstance(boundary_condition, tuple):
            if len(boundary_condition) != self._dim:
                raise ValueError(
                    "size mismatch, "
                    f"`boundary_condition`: {len(boundary_condition)}, `size`: {self._dim}."
                    "The length of `boundary_condition` must be the same as that of size."
                )

        self._boundary_condition = boundary_condition

        graph: PyGraph = PyGraph(multigraph=False)
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

    def _coordinate_to_index(self, coord: np.ndarray) -> int:
        """Convert the coordinate of a lattice point to an integer for labeling.

            When self.size=(l0, l1), then a coordinate (x0, x1) is converted as
            x0 + x1*l0.

        Args:
            coord: Input coordinate to be converted.

        Returns:
            Return x0 + x1*l0 where coord=np.array([x0, x1]) and self.size=(l0, l1).
        """
        base = np.array([np.prod(self._size[:i]) for i in range(self._dim)], dtype=int)
        return np.dot(coord, base).item()

    def _self_loops(self) -> list[tuple[int, int, complex]]:
        """Return a list consisting of the self-loops on all the nodes.

        Returns:
            A list of the self-loops.
        """
        onsite_parameter = self._onsite_parameter
        num_nodes = self._num_sites_per_cell * np.prod(self._size)
        return [(node_a, node_a, onsite_parameter) for node_a in range(num_nodes)]

    def _bulk_edges(self) -> list[tuple[int, int, complex]]:
        """Return a list consisting of the edges in the bulk, which don't cross the boundaries.

        Returns:
            A list of weighted edges that do not cross the boundaries.
        """
        edge_parameter = self._edge_parameter
        num_sites_per_cell = self._num_sites_per_cell
        list_of_edges = []
        unit_cell_coordinates = list(product(*map(range, self._size)))

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
            if x != self._rows - 1:
                cell_b_idx = self._coordinate_to_index(np.array([x, y]) + np.array([1, 0]))
                cell_b_0 = num_sites_per_cell * cell_b_idx
                list_of_edges.append((cell_a_1, cell_b_0, edge_parameter))

            # one cell north if not at the north boundary
            if y != self._cols - 1:
                cell_b_idx = self._coordinate_to_index(np.array([x, y]) + np.array([0, 1]))
                cell_b_0 = num_sites_per_cell * cell_b_idx
                list_of_edges.append((cell_a_2, cell_b_0, edge_parameter))

            # one cell west and north if not at west north boundary
            if x != 0 and y != self._cols - 1:
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
            A list of weighted edges that cross the boundaries.
        """
        list_of_edges = []
        edge_parameter = self._edge_parameter
        num_sites_per_cell = self._num_sites_per_cell
        boundary_condition = self._boundary_condition

        is_x_periodic = boundary_condition[0] == BoundaryCondition.PERIODIC
        is_y_periodic = boundary_condition[1] == BoundaryCondition.PERIODIC

        # add edges when the boundary condition is periodic.
        # The periodic boundary condition in the x direction.
        # It makes sense only when rows is greater than 1.
        if is_x_periodic and self._rows > 1:
            for y in range(self._cols):
                cell_a_idx = self._coordinate_to_index(np.array([self._rows - 1, y]))
                cell_a_1 = num_sites_per_cell * cell_a_idx + 1

                cell_b_idx = self._coordinate_to_index(np.array([0, y]))
                cell_b_0 = num_sites_per_cell * cell_b_idx

                list_of_edges.append((cell_a_1, cell_b_0, edge_parameter.conjugate()))

        # The periodic boundary condition in the y direction.
        # It makes sense only when cols is greater than 1.
        if is_y_periodic and self._cols > 1:
            for x in range(self._rows):
                cell_a_idx = self._coordinate_to_index(np.array([x, self._cols - 1]))
                cell_a_2 = num_sites_per_cell * cell_a_idx + 2

                cell_b_idx = self._coordinate_to_index(np.array([x, 0]))
                cell_b_0 = num_sites_per_cell * cell_b_idx

                list_of_edges.append((cell_a_2, cell_b_0, edge_parameter.conjugate()))

        if is_x_periodic and is_y_periodic:
            # The periodic boundary condition in the diagonal directions.
            for x in range(1, self._rows):
                cell_a_idx = self._coordinate_to_index(np.array([x, self._cols - 1]))
                cell_a_2 = num_sites_per_cell * cell_a_idx + 2

                cell_b_idx = self._coordinate_to_index(np.array([(x - 1) % self._rows, 0]))
                cell_b_1 = num_sites_per_cell * cell_b_idx + 1

                list_of_edges.append((cell_a_2, cell_b_1, edge_parameter.conjugate()))

            for y in range(self._cols - 1):
                cell_a_idx = self._coordinate_to_index(np.array([0, y]))
                cell_a_2 = num_sites_per_cell * cell_a_idx + 2

                cell_b_idx = self._coordinate_to_index(
                    np.array([self._rows - 1, (y + 1) % self._cols])
                )
                cell_b_1 = num_sites_per_cell * cell_b_idx + 1

                list_of_edges.append((cell_a_2, cell_b_1, edge_parameter.conjugate()))

            # isolating x = 0, y = cols - 1 to prevent duplicating edges
            cell_a_idx = self._coordinate_to_index(np.array([0, self._cols - 1]))
            cell_a_2 = num_sites_per_cell * cell_a_idx + 2

            cell_b_idx = self._coordinate_to_index(np.array([self._rows - 1, 0]))
            cell_b_1 = num_sites_per_cell * cell_b_idx + 1

            list_of_edges.append((cell_a_2, cell_b_1, edge_parameter.conjugate()))

        for i in range(self._dim):
            if not isinstance(boundary_condition[i], BoundaryCondition):
                raise ValueError(
                    f"Invalid `boundary condition` {boundary_condition[i]} is given."
                    "`boundary condition` must be "
                    + " or ".join(str(bc) for bc in BoundaryCondition)
                )
        return list_of_edges

    def _default_position(self) -> dict[int, list[float]]:
        """Return a dictionary of default positions for visualization of a two-dimensional lattice.

        Returns:
            The keys are the labels of lattice points,
            and the values are two-dimensional coordinates.
        """
        boundary_condition = self._boundary_condition
        num_sites_per_cell = self._num_sites_per_cell
        pos = {}
        width = np.array([0.0, 0.0])
        for i in (0, 1):
            if boundary_condition[i] == BoundaryCondition.PERIODIC:
                # the positions are shifted along the y-direction
                # when the boundary condition in the x-direction is periodic and vice versa.
                # The width of the shift is fixed to 0.2.
                width[(i + 1) % 2] = 0.2
        for cell_idx in range(np.prod(self._size)):
            # maps an cell index to two-dimensional coordinate
            # the positions are shifted so that the edges between boundaries can be seen
            # for the periodic cases.
            cell_coord = np.array(divmod(cell_idx, self._size[0])[::-1]) + width * np.cos(
                np.pi
                * (np.array(divmod(cell_idx, self._size[0])))
                / (np.array(self._size)[::-1] - 1)
            )

            for i in range(num_sites_per_cell):
                node_i = num_sites_per_cell * cell_idx + i
                pos[node_i] = (np.dot(cell_coord, self._basis) + self._cell_positions[i]).tolist()
        return pos

    def _style_pos(self) -> dict[int, list[float]]:
        """Return a dictionary of positions for visualization of a two-dimensional lattice without
            boundaries.

        Returns:
           The keys are the labels of lattice points,
           and the values are two-dimensional coordinates.
        """
        num_sites_per_cell = self._num_sites_per_cell
        basis = self._basis
        cell_positions = self._cell_positions
        pos = {}

        for cell_idx in range(np.prod(self._size)):
            # maps an cell index to two-dimensional coordinate
            # the positions are shifted so that the edges between boundaries can be seen
            # for the periodic cases.
            cell_coord = np.array(divmod(cell_idx, self._size[0])[::-1])

            for i in range(num_sites_per_cell):
                node_i = num_sites_per_cell * cell_idx + i
                pos[node_i] = (np.dot(cell_coord, basis) + cell_positions[i]).tolist()
        return pos

    @property
    def rows(self) -> int:
        """Number of unit cells in the x direction.

        Returns:
            The number of rows of the lattice.
        """
        return self._rows

    @property
    def cols(self) -> int:
        """Number of unit cells in the y direction.

        Returns:
            The number of columns of the lattice.
        """
        return self._cols

    @property
    def size(self) -> tuple[int, int]:
        """Number of unit cells in the x and y direction, respectively.

        Returns:
            The size of the lattice.
        """
        return self._size

    @property
    def edge_parameter(self) -> complex:
        """Weights on all edges.

        Returns:
            The parameter for the edges.
        """
        return self._edge_parameter

    @property
    def onsite_parameter(self) -> complex:
        """Weight on the self-loops.

        Returns:
            The parameter for the self-loops.
        """
        return self._onsite_parameter

    @property
    def boundary_condition(self) -> BoundaryCondition | tuple[BoundaryCondition, BoundaryCondition]:
        """Boundary condition for the entire lattice.

        Returns:
            The boundary condition.
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
                https://www.rustworkx.org/apiref/rustworkx.visualization.mpl_draw.html
                for details.
        """
        graph = self.graph

        if style is None:
            style = LatticeDrawStyle()
        elif not isinstance(style, LatticeDrawStyle):
            style = LatticeDrawStyle(**style)

        if style.pos is None:
            style.pos = self._default_position()

        graph.remove_edges_from(self.boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
