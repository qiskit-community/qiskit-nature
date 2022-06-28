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

"""The hyper-cubic lattice"""
from dataclasses import asdict
from itertools import product
from math import pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from retworkx import PyGraph

from .boundary_condition import BoundaryCondition
from .lattice import Lattice, LatticeDrawStyle


class HyperCubicLattice(Lattice):
    """Hyper-cubic lattice in d dimensions.

    The :class:`HyperCubicLattice` can be initialized with
    tuples of `size`, `edge_parameters`, and `boundary_conditions`.
    For example,

    .. jupyter-execute::

        from qiskit_nature.second_quantization.problems.lattice import (
            BoundaryCondition,
            HyperCubicLattice,
            )

        lattice = HyperCubicLattice(
            size = (3, 4, 5),
            edge_parameter = (1.0, -2.0, 3.0),
            onsite_parameter = 2.0,
            boundary_condition = (BoundaryCondition.OPEN, BoundaryCondition.OPEN, BoundaryCondition.OPEN)
            )

    is a three-dimensional lattice of size 3 by 4 by 5, which has weights 1.0, -2.0, 3.0 on edges
    in x, y, and z directions, respectively, and weights 2.0 on self-loops.
    The boundary conditions are open for all the directions.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        edge_parameter: Union[complex, Tuple[complex, ...]] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: Union[
            BoundaryCondition, Tuple[BoundaryCondition, ...]
        ] = BoundaryCondition.OPEN,
    ) -> None:
        """
        Args:
            size: Lengths of each dimension.
            edge_parameter: Weights on the edges in each direction.
                When it is a single value, it is interpreted as a tuple of the same length as `size`
                consisting of the same values.
                Defaults to 1.0.
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                This is uniform over the lattice points.
                Defaults to 0.0.
            boundary_condition: Boundary condition for each dimension.
                The available boundary conditions are:
                BoundaryCondition.OPEN, BoundaryCondition.PERIODIC.
                When it is a single value, it is interpreted as a tuple of the same length as `size`
                consisting of the same values.
                Defaults to BoundaryCondition.OPEN.

        Raises:
            ValueError: When edge parameter or boundary condition is a tuple,
                the length of that is not the same as that of size.
        """

        self._dim = len(size)
        self._size = size

        # edge parameter
        if isinstance(edge_parameter, (int, float, complex)):
            edge_parameter = (edge_parameter,) * self._dim
        elif isinstance(edge_parameter, tuple):
            if len(edge_parameter) != self._dim:
                raise ValueError(
                    "size mismatch, "
                    f"`edge_parameter`: {len(edge_parameter)}, `size`: {self._dim}."
                    "The length of `edge_parameter` must be the same as that of size."
                )

        self._edge_parameter = edge_parameter

        self._onsite_parameter = onsite_parameter

        # boundary condition
        if isinstance(boundary_condition, BoundaryCondition):
            boundary_condition = (boundary_condition,) * self._dim
        elif isinstance(boundary_condition, tuple):
            if len(boundary_condition) != self._dim:
                raise ValueError(
                    "size mismatch, "
                    f"`boundary_condition`: {len(boundary_condition)}, `size`: {self._dim}."
                    "The length of `boundary_condition` must be the same as that of size."
                )

        self._boundary_condition = boundary_condition

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(np.prod(size)))

        # add edges excluding the boundary edges
        bulk_edge_list = self._bulk_edges()
        graph.add_edges_from(bulk_edge_list)

        # add self-loops.
        self_loop_list = self._self_loops()
        graph.add_edges_from(self_loop_list)

        # add edges that cross the boundaries
        boundary_edge_list = self._create_boundary_edges()
        graph.add_edges_from(boundary_edge_list)

        # a list of edges that depend on the boundary condition
        self._boundary_edges = [(edge[0], edge[1]) for edge in boundary_edge_list]

        super().__init__(graph)

        # default position for one and two-dimensional cases.
        self.pos = self._default_position()

    @property
    def dim(self) -> int:
        """Dimensions of the hyper cubic lattice.

        Returns:
            the dimension.
        """
        return self._dim

    @property
    def size(self) -> Tuple[int, ...]:
        """Lengths of each dimension.

        Returns:
            the size.
        """
        return self._size

    @property
    def edge_parameter(self) -> Union[complex, Tuple[complex, ...]]:
        """Weights on the edges in each direction.

        Returns:
            the parameter for the edges.
        """
        return self._edge_parameter

    @property
    def onsite_parameter(self) -> complex:
        """Weight on the self-loops

        Returns:
            the parameter for the self-loops.
        """
        return self._onsite_parameter

    @property
    def boundary_condition(self) -> Union[BoundaryCondition, Tuple[BoundaryCondition, ...]]:
        """Boundary condition for each dimension.

        Returns:
            the boundary condition.
        """
        return self._boundary_condition

    def _coordinate_to_index(self, coord: np.ndarray) -> int:
        """Convert the coordinate of a lattice point to an integer for labeling.
            When size=(l0, l1, l2, ...), then a coordinate (x0, x1, x2, ...) is converted as
            x0 + x1*l0 + x2*l0*l1 + ... .
        Args:
            coord: Input coordinate to be converted.

        Returns:
            Return x0 + x1*l0 + x2*l0*l1 + ...
            when coord=np.array([x0, x1, x2...]) and size=(l0, l1, l2, ...).
        """
        size = self._size
        dim = len(size)
        base = np.array([np.prod(size[:i]) for i in range(dim)], dtype=int)
        return np.dot(coord, base).item()

    def _self_loops(self) -> List[Tuple[int, int, complex]]:
        """Return a list consisting of the self-loops on all the nodes.
        Returns:
            List[Tuple[int, int, complex]] : List of the self-loops.
        """
        num_nodes = np.prod(self._size)
        return [(node_a, node_a, self._onsite_parameter) for node_a in range(num_nodes)]

    def _bulk_edges(self) -> List[Tuple[int, int, complex]]:
        """Return a list consisting of the edges in the bulk, which don't cross the boundaries.

        Returns:
            List[Tuple[int, int, complex]] : List of weighted edges that don't cross the boundaries.
        """
        list_of_edges = []
        size = self._size
        edge_parameter = self._edge_parameter
        dim = len(size)
        coordinates = list(product(*map(range, size)))
        # add edges excluding the boundary edges
        for coord in np.array(coordinates):
            for i in range(dim):
                if coord[i] != size[i] - 1:
                    relative_vector = np.eye(dim, dtype=int)[i]
                    node_a = self._coordinate_to_index(coord)
                    node_b = self._coordinate_to_index(coord + relative_vector)
                    list_of_edges.append((node_a, node_b, edge_parameter[i]))
        return list_of_edges

    def _create_boundary_edges(self) -> List[Tuple[int, int, complex]]:
        """Return a list consisting of the edges that cross the boundaries
            depending on the boundary conditions.

        Raises:
            ValueError: Given boundary condition is invalid values.
        Returns:
            List[Tuple[int, int, complex]]: List of weighted edges that cross the boundaries.
        """
        list_of_edges = []
        size = self._size
        edge_parameter = self._edge_parameter
        boundary_condition = self._boundary_condition
        dim = len(size)
        for i in range(dim):
            # add edges when the boundary condition is periodic.
            # when the boundary condition in the i-th direction is periodic,
            # it makes sense only when size[i] is greater than 2.
            if boundary_condition[i] == BoundaryCondition.PERIODIC:
                if size[i] <= 2:
                    continue
                size_list = list(size)
                size_list[i] = 1
                coordinates = list(product(*map(range, size_list)))
                relative_vector = np.eye(dim, dtype=int)[i]
                for coord in np.array(coordinates):
                    node_b = self._coordinate_to_index(coord)
                    node_a = self._coordinate_to_index((coord - relative_vector) % size)
                    list_of_edges.append((node_b, node_a, edge_parameter[i].conjugate()))
            elif boundary_condition[i] == BoundaryCondition.OPEN:
                continue
            else:
                raise ValueError(
                    f"Invalid `boundary condition` {boundary_condition[i]} is given."
                    "`boundary condition` must be "
                    + " or ".join(str(bc) for bc in BoundaryCondition)
                )
        return list_of_edges

    def _default_position(self) -> Optional[Dict[int, List[float]]]:
        """Return a dictionary of default positions for visualization of
            a one- or two-dimensional lattice.

        Returns:
            Optional[Dict[int, List[float]]]: The keys are the labels of lattice points,
                and the values are two-dimensional coordinates.
                When the dimension is larger than 2, it returns None.
        """
        size = self._size
        boundary_condition = self._boundary_condition
        dim = len(size)
        if dim == 1:
            if boundary_condition[0] == BoundaryCondition.OPEN:
                pos = {i: [float(i), 0.0] for i in range(size[0])}
            elif boundary_condition[0] == BoundaryCondition.PERIODIC:
                theta = 2 * pi / size[0]
                pos = {i: [np.cos(i * theta), np.sin(i * theta)] for i in range(size[0])}
        elif dim == 2:
            pos = {}
            width = np.array([0.0, 0.0])
            for i in (0, 1):
                if boundary_condition[i] == BoundaryCondition.PERIODIC:
                    # the positions are shifted along the y-direction
                    # when the boundary condition in the x-direction is periodic and vice versa.
                    # The width of the shift is fixed to 0.2.
                    width[(i + 1) % 2] = 0.2
            for index in range(np.prod(size)):
                # maps an index to two-dimensional coordinate
                # the positions are shifted so that the edges between boundaries can be seen
                # for the periodic cases.
                coord = np.array(divmod(index, size[0]))[::-1] + width * np.sin(
                    pi * np.array(divmod(index, size[0])) / (np.array(size)[::-1] - 1)
                )
                pos[index] = coord.tolist()
        else:
            pos = None

        return pos

    def draw_without_boundary(
        self,
        self_loop: bool = False,
        style: Optional[LatticeDrawStyle] = None,
    ):
        r"""Draw the lattice with no edges between the boundaries.

        Args:
            self_loop: Draw self-loops in the lattice. Defaults to False.
            style : Styles for retworkx.visualization.mpl_draw.
                Please see
                https://qiskit.org/documentation/retworkx/stubs/retworkx.visualization.mpl_draw.html#retworkx.visualization.mpl_draw
                for details.
        """
        graph = self.graph

        if style is None:
            style = LatticeDrawStyle()
        elif not isinstance(style, LatticeDrawStyle):
            style = LatticeDrawStyle(**style)

        if style.pos is None:
            if self.dim == 1:
                style.pos = {i: [i, 0] for i in range(self.size[0])}
            elif self.dim == 2:
                style.pos = {
                    i: [i % self.size[0], i // self.size[0]] for i in range(np.prod(self.size))
                }

        graph.remove_edges_from(self._boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
