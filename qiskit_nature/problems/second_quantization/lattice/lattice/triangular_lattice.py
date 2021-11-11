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

"""The triangular lattice"""
from dataclasses import asdict
from itertools import product
from math import pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from retworkx import PyGraph

from .lattice import DrawStyle, Lattice


def _coordinate_to_index(coord: np.ndarray, size: Tuple[int, int]) -> int:
    """Convert the coordinate of a lattice point to an integer for labeling.
        When size=(l0, l1), then a coordinate (x0, x1) is converted as
        x0 + x1*l0.
    Args:
        coord: Input coordinate to be converted.
        size: Lengths of each dimension.

    Returns:
        int: Return x0 + x1*l0 when coord=np.array([x0, x1]) and size=(l0, l1).
    """
    dim = 2
    base = np.array([np.prod(size[:i]) for i in range(dim)], dtype=int)
    return np.dot(coord, base).item()


def _self_loops(size: Tuple[int, int], onsite_parameter: complex) -> List[Tuple[int, int, complex]]:
    """Return a list consisting of the self-loops on all the nodes.
    Args:
        size : Lengths of each dimension.
        onsite_parameter: Weight of the self-loops.
    Returns:
        List[Tuple[int, int, complex]] : List of the self-loops.
    """
    num_nodes = np.prod(size)
    return [(node_a, node_a, onsite_parameter) for node_a in range(num_nodes)]


def _bulk_edges(
    size: Tuple[int, int], edge_parameter: Tuple[complex, complex, complex]
) -> List[Tuple[int, int, complex]]:
    """Return a list consisting of the edges in th bulk, which don't cross the boundaries.

    Args:
        size : Lengths of each dimension.
        edge_parameter : Weights on the edges in the x, y, and diagonal directions.

    Returns:
        List[Tuple[int, int, complex]] : List of weighted edges that don't cross the boundaries.
    """
    list_of_edges = []
    rows, cols = size
    coordinates = list(product(*map(range, size)))
    for x, y in coordinates:
        node_a = _coordinate_to_index(np.array([x, y]), size)
        for i in range(3):
            # x direction
            if i == 0 and x != rows - 1:
                node_b = _coordinate_to_index(np.array([x, y]) + np.array([1, 0]), size)
            # y direction
            elif i == 1 and y != cols - 1:
                node_b = _coordinate_to_index(np.array([x, y]) + np.array([0, 1]), size)
            # diagonal direction
            elif i == 2 and x != rows - 1 and y != cols - 1:
                node_b = _coordinate_to_index(np.array([x, y]) + np.array([1, 1]), size)
            else:
                continue
            list_of_edges.append((node_a, node_b, edge_parameter[i]))
    return list_of_edges


def _boundary_edges(
    size: Tuple[int, int], edge_parameter: Tuple[complex, complex, complex], boundary_condition: str
) -> List[Tuple[int, int, complex]]:
    """Return a list consisting of the edges that cross the boundaries
        depending on the boundary conditions.

    Args:
        size : Lengths of each dimension.
        edge_parameter : Weights on the edges in each direction.
        boundary_condition : Boundary condition for each dimension.
            The available boundary conditions are: "open", "periodic".
    Raises:
        ValueError: Given boundary condition is invalid values.
    Returns:
        List[Tuple[int, int, complex]]: List of weighted edges that cross the boundaries.
    """
    list_of_edges = []
    rows, cols = size
    # add edges when the boundary condition is periodic.
    if boundary_condition == "periodic":
        # The periodic boundary condition in the x direction.
        # It makes sense only when rows is greater than 2.
        if rows > 2:
            for y in range(cols):
                node_a = (y + 1) * rows - 1
                node_b = node_a - (rows - 1)  # node_b < node_a
                list_of_edges.append((node_b, node_a, edge_parameter[0].conjugate()))
        # The periodic boundary condition in the y direction.
        # It makes sense only when cols is greater than 2.
        if cols > 2:
            for x in range(rows):
                node_a = rows * (cols - 1) + x
                node_b = x  # node_b < node_a
                list_of_edges.append((node_b, node_a, edge_parameter[1].conjugate()))
        # The periodic boundary condition in the diagonal direction.
        for y in range(cols - 1):
            node_a = (y + 1) * rows - 1
            node_b = node_a + 1  # node_b > node_a
            list_of_edges.append((node_a, node_b, edge_parameter[2]))

        for x in range(rows - 1):
            node_a = rows * (cols - 1) + x
            node_b = x + 1  # node_b < node_a
            list_of_edges.append((node_b, node_a, edge_parameter[2].conjugate()))

        node_a = rows * cols - 1
        node_b = 0  # node_b < node_a
        list_of_edges.append((node_b, node_a, edge_parameter[2].conjugate()))

    elif boundary_condition != "open":
        raise ValueError(
            f"Invalid `boundary condition` {boundary_condition} is given."
            "`boundary condition` must be `open` or `periodic`."
        )
    return list_of_edges


def _default_position(size: Tuple[int, int], boundary_condition: str) -> Dict[int, List[float]]:
    """return a dictionary of default positions for visualization of a two-dimensional lattice.

    Args:
        size : Lengths of each dimension.
        boundary_condition : Boundary condition for the lattice.
                The available boundary conditions are: "open", "periodic".
    Returns:
        Dict[int, List[float]] : The keys are the labels of lattice points,
            and the values are two-dimensional coordinates.
    """
    pos = {}
    width = 0.0
    if boundary_condition == "periodic":
        # the positions are shifted along the x- and y-direction
        # when the boundary condition is periodic.
        # The width of the shift is fixed to 0.2.
        width = 0.2
    for index in range(np.prod(size)):
        # maps an index to two-dimensional coordinate
        # the positions are shifted so that the edges between boundaries can be seen
        # for the periodic cases.
        coord = np.array(divmod(index, size[0]))[::-1] + width * np.sin(
            pi * np.array(divmod(index, size[0])) / (np.array(size)[::-1] - 1)
        )
        pos[index] = coord.tolist()
    return pos


class TriangularLattice(Lattice):
    """Triangular lattice."""

    def __init__(
        self,
        rows: int,
        cols: int,
        edge_parameter: Union[complex, Tuple[complex, complex, complex]] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: str = "open",
    ) -> None:
        """
        Args:
            rows: Length of the x direction.
            cols: Length of the y direction.
            edge_parameter: Weights on the edges in x, y and diagonal directions.
                This is specified as a tuple of length 3 or a single value.
                When it is a single value, it is interpreted as a tuple of length 3
                consisting of the same values.
                Defaults to 1.0,
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
            boundary_condition: Boundary condition for the lattice.
                The available boundary conditions are: "open", "periodic".
                Defaults to "open".

        Raises:
            ValueError: Given size, edge parameter or boundary condition are invalid values.
        """
        self.rows = rows
        self.cols = cols
        self.size = (rows, cols)
        self.dim = 2
        self.boundary_condition = boundary_condition

        if rows < 2 or cols < 2 or (rows, cols) == (2, 2):
            # If it's True, triangular lattice can't be well defined.
            raise ValueError(
                "Both of `rows` and `cols` must not be (2, 2)"
                "and must be greater than or equal to 2."
            )

        if isinstance(edge_parameter, (int, float, complex)):
            edge_parameter = (edge_parameter, edge_parameter, edge_parameter)
        elif isinstance(edge_parameter, tuple):
            if len(edge_parameter) != 3:
                raise ValueError(
                    f"The length of `edge_parameter` must be 3, not {len(edge_parameter)}."
                )

        self.edge_parameter = edge_parameter
        self.onsite_parameter = onsite_parameter

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(np.prod(self.size)))

        # add edges excluding the boundary edges
        bulk_edges = _bulk_edges(self.size, self.edge_parameter)
        graph.add_edges_from(bulk_edges)

        # add self-loops
        self_loop_list = _self_loops(self.size, self.onsite_parameter)
        graph.add_edges_from(self_loop_list)

        # add edges that cross the boundaries
        boundary_edge_list = _boundary_edges(
            self.size, self.edge_parameter, self.boundary_condition
        )
        graph.add_edges_from(boundary_edge_list)

        # a list of edges that depend on the boundary condition
        self.boundary_edges = [(edge[0], edge[1]) for edge in boundary_edge_list]
        super().__init__(graph)
        # default position
        self.pos = _default_position(self.size, self.boundary_condition)

    def draw_without_boundary(
        self,
        self_loop: bool = False,
        style: Optional[DrawStyle] = None,
    ):
        r"""Draw the lattice with no edges between the boundaries.

        Args:
            self_loop: Draw self-loops in the lattice. Defaults to False.
            style : Styles for retworkx.visualization.mpl_draw.
                Please see
                https://qiskit.org/documentation/retworkx/stubs/retworkx.visualization.mpl_draw.html#retworkx.visualization.mpl_draw
                for details.

        Returns:
            A matplotlib figure for the visualization if not running with an
            interactive backend (like in jupyter) or if ``ax`` is not set.
        """
        graph = self.graph

        if style is None:
            style = DrawStyle()
        elif not isinstance(style, DrawStyle):
            style = DrawStyle(**style)

        if style.pos is None:
            if self.dim == 1:
                style.pos = {i: [i, 0] for i in range(self.size[0])}
            elif self.dim == 2:
                style.pos = {
                    i: [i % self.size[0], i // self.size[0]] for i in range(np.prod(self.size))
                }

        graph.remove_edges_from(self.boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
