# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The hexagonal lattice"""
from dataclasses import asdict
from itertools import product
from math import pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rustworkx import PyGraph, generators

from .lattice import LatticeDrawStyle, Lattice
from .boundary_condition import BoundaryCondition
from rustworkx.visualization import graphviz_draw


class HexagonalLattice(Lattice):
    """Hexagonal lattice."""

    def __init__(
        self,
        num_rings: int,
        edge_parameter: complex = 1.0,
        onsite_parameter: complex = 0.0,
        # boundary_condition: BoundaryCondition
        # heavy: bool
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

        Raises:
            ValueError: Given size or edge parameter are invalid values.
        """
        self.edge_parameter = edge_parameter
        if num_rings == 1:
            self.size = 6
        elif num_rings == 2:
            self.size = 10
        else:
            raise NotImplementedError("sad")

        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(self.size))

        edges = []
        for node in range(self.size):
            edges.append((node, (node+1)%self.size, self.edge_parameter))

        graph.add_edges_from(edges)

        # G = generators.heavy_hex_graph(5)
        G = generators.heavy_hex_graph(7)
# set data payload to index
        for node in G.node_indices():
            G[node] = node

        # # question? what is this and do we need it?
        # # add edges excluding the boundary edges
        # bulk_edges = self._bulk_edges()
        # graph.add_edges_from(bulk_edges)

        # #question - what happening in triangular _self_loops func?
        # # add self-loops
        # self_loop_list = self._self_loops()
        # graph.add_edges_from(self_loop_list)

        # # add edges that cross the boundaries
        # boundary_edge_list = self._boundary_edges()
        # graph.add_edges_from(boundary_edge_list)

        super().__init__(G)
        # default position
        # self.pos = self._default_position()