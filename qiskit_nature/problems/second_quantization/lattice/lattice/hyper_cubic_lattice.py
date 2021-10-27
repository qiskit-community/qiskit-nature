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
from itertools import product
from math import pi
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from retworkx import PyGraph

from .lattice import Lattice, _add_draw_signature


class HyperCubicLattice(Lattice):
    """Hyper-cubic lattice in d dimension.

    The :class:`HyperCubicLattice` can be initialized with
    tuples of `size`, `edge_parameters`, and `boundary_conditions`.
    For example,

    .. jupyter-execute::

        from qiskit_nature.problems.second_quantization.lattice import HyperCubicLattice

        lattice = HyperCubicLattice(
            size = (3, 4, 5),
            edge_parameter = (1.0, -2.0, 3.0),
            onsite_parameter = 2.0,
            boundary_condition = ("open", "open", "open")
            )

    is a three-dimensional lattice of size 3 by 4 by 5, which has weights 1.0, -2.0, 3.0 on edges
    in x, y, and z directions, respectively, and weights 2.0 on self-loops.
    The boundary conditions are "open" for all the directions.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        edge_parameter: Union[complex, Tuple[complex, ...]] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: Union[str, Tuple[str, ...]] = "open",
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
                The available boundary conditions are: "open", "periodic".
                When it is a single value, it is interpreted as a tuple of the same length as `size`
                consisting of the same values.
                Defaults to "open".

        Raises:
            ValueError: Given edge parameter or boundary condition are invalid values.
            TypeError: When edge parameter is a tuple,
                the length of edge parameter is not the same as that of size.
        """

        self.dim = len(size)

        self.size = size

        # edge parameter
        if isinstance(edge_parameter, (int, float, complex)):
            edge_parameter = (edge_parameter,) * self.dim
        elif isinstance(edge_parameter, tuple):
            if len(edge_parameter) == self.dim:
                pass
            else:
                raise TypeError(
                    "size mismatch, "
                    f"`edge_parameter`: {len(edge_parameter)}, `size`: {self.dim}."
                    "The length of `edge_parameter` must be the same as that of size."
                )

        self.edge_parameter = edge_parameter
        # onsite parameter
        self.onsite_parameter = onsite_parameter

        # boundary condition
        if isinstance(boundary_condition, str):
            boundary_condition = (boundary_condition,) * self.dim
        elif isinstance(boundary_condition, tuple):
            if len(boundary_condition) != self.dim:
                raise ValueError(
                    f"The length of `boundary_condition` must be the same as that of size, {self.dim}."
                )

        self.boundary_conditions = boundary_condition

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(np.prod(size)))

        # add edges excluding the boundary edges
        coordinates = list(product(*map(range, size)))
        base = np.array([np.prod(size[:i]) for i in range(self.dim)], dtype=int)
        for coord in coordinates:
            for i in range(self.dim):
                if coord[i] != size[i] - 1:
                    node_a = np.dot(coord, base)
                    node_b = node_a + base[i]
                    graph.add_edge(node_a, node_b, edge_parameter[i])

        # add self-loops
        for node_a in range(np.prod(size)):
            graph.add_edge(node_a, node_a, onsite_parameter)

        # depend on the boundary condition
        self.boundary_edges = []
        for i in range(self.dim):
            # add edges when the boundary condition is periodic.
            # when the boundary condition in the i-th direction is periodic,
            # it makes sense only when size[i] is greater than 2.
            if boundary_condition[i] == "periodic":
                if size[i] <= 2:
                    continue
                size_list = list(size)
                size_list[i] = 1
                coordinates = list(product(*map(range, size_list)))
                for coord in coordinates:
                    node_b = np.dot(coord, base)
                    node_a = node_b + base[i] * (size[i] - 1)
                    if node_a < node_b:
                        graph.add_edge(node_a, node_b, edge_parameter[i])
                        self.boundary_edges.append((node_a, node_b))
                    elif node_a > node_b:
                        graph.add_edge(node_b, node_a, edge_parameter[i].conjugate())
                        self.boundary_edges.append((node_a, node_b))
            elif boundary_condition[i] != "open":
                raise ValueError(
                    f"Invalid `boundary condition` {boundary_condition[i]} is given."
                    "`boundary condition` must be `open` or `periodic`."
                )

        super().__init__(graph)

        # default position for one and two-dimensional cases.
        if self.dim == 1:
            if self.boundary_conditions[0] == "open":
                self.pos = {i: [i, 0] for i in range(self.size[0])}
            elif self.boundary_conditions[0] == "periodic":
                theta = 2 * pi / self.size[0]
                self.pos = {i: [np.cos(i * theta), np.sin(i * theta)] for i in range(self.size[0])}
        elif self.dim == 2:
            self.pos = {}
            for index in range(np.prod(self.size)):
                # maps an index to two-dimensional coordinate
                # the positions are shifted so that the edges between boundaries can be seen
                # for the periodic cases.
                x = index % self.size[0]
                y = index // self.size[0]
                if self.boundary_conditions[1] == "open":
                    return_x = x
                elif self.boundary_conditions[1] == "periodic":
                    return_x = x + 0.2 * np.sin(pi * y / (self.size[1] - 1))
                if self.boundary_conditions[0] == "open":
                    return_y = y
                elif self.boundary_conditions[0] == "periodic":
                    return_y = y + 0.2 * np.sin(pi * x / (self.size[0] - 1))
                self.pos[index] = [return_x, return_y]

    @_add_draw_signature
    def draw_without_boundary(
        self,
        self_loop: bool = False,
        **kwargs,
    ):
        r"""Draw the lattice with no edges between the boundaries.

        Args:
            self_loop: Draw self-loops in the lattice. Defaults to False.
            **kwargs : Kwargs for retworkx.visualization.mpl_draw.
                Please see
                https://qiskit.org/documentation/retworkx/stubs/retworkx.visualization.mpl_draw.html#retworkx.visualization.mpl_draw
                for details.

        Returns:
            A matplotlib figure for the visualization if not running with an
            interactive backend (like in jupyter) or if ``ax`` is not set.
        """
        graph = self.graph

        if "pos" not in kwargs:
            if self.dim == 1:
                kwargs["pos"] = {i: [i, 0] for i in range(self.size[0])}
            elif self.dim == 2:
                kwargs["pos"] = {
                    i: [i % self.size[0], i // self.size[0]] for i in range(np.prod(self.size))
                }

        graph.remove_edges_from(self.boundary_edges)

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **kwargs,
        )
