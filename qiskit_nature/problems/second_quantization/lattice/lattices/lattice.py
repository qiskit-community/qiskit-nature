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

"""General Lattice."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numbers

import numpy as np

try:
    import networkx as nx

    HAS_NX = True
except ImportError:
    HAS_NX = False

from retworkx import NodeIndices, PyGraph, WeightedEdgeList, adjacency_matrix, networkx_converter
from retworkx.visualization import mpl_draw

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.tools.visualization import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    # pylint: disable=unused-import
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap


@dataclass
class LatticeDrawStyle:
    """A stylesheet for lattice figure.
    Please see
    https://qiskit.org/documentation/retworkx/stubs/retworkx.visualization.mpl_draw.html#retworkx.visualization.mpl_draw
    for each element.
    """

    # position
    pos: Optional[dict] = None

    # Matplotlib Axes object
    ax: Optional["Axes"] = None  # pylint:disable=invalid-name

    with_labels: bool = False

    node_list: Optional[list] = None

    edge_list: Optional[list] = None

    node_size: Union[int, list] = 300

    node_color: Union[
        str,
        Tuple[float, float, float],
        Tuple[float, float, float],
        List[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]],
    ] = "#1f78b4"

    node_shape: str = "o"

    # node and edge transparency
    alpha: Optional[float] = None

    # Matplotlib colormap object
    cmap: Optional["Colormap"] = None

    # minimum value for node colormap scaling
    vmin: Optional[float] = None

    # minimum value for node colormap scaling
    vmax: Optional[float] = None

    linewidths: Union[float, Sequence] = 1.0

    width: Union[float, Sequence] = 1.0

    edge_color: Union[str, Sequence] = "k"

    edge_cmap: Optional["Colormap"] = None

    # minimum value for edge colormap scaling
    edge_vmin: Optional[float] = None

    # maximum value for node colormap scaling
    edge_vmax: Optional[float] = None

    style: str = "solid"

    labels: Optional[Callable] = None

    edge_labels: Optional[Callable] = None

    font_size: int = 12

    font_color: str = "k"

    font_weight: str = "normal"

    font_family: str = "sans-serif"

    label: Optional[str] = None

    connectionstyle: str = "arc3"


class Lattice:
    """General Lattice."""

    def __init__(self, graph: Union[PyGraph, nx.Graph]) -> None:
        """
        Args:
            graph: Input graph for Lattice. Can be provided as ``retworkx.PyGraph``, which is
                used internally, or, for convenience, as ``networkx.Graph``. The graph
                cannot be a multigraph.

        Raises:
            ValueError: If the input graph is a multigraph.
            ValueError: If the graph edges are non-numeric.
        """
        if HAS_NX and isinstance(graph, nx.Graph):
            graph = networkx_converter(graph)

        if graph.multigraph:
            raise ValueError(
                f"Invalid `graph.multigraph` {graph.multigraph} is given. "
                "`graph.multigraph` must be `False`."
            )

        # validate the edge weights
        for edge_index, edge in graph.edge_index_map().items():
            weight = edge[2]
            if weight is None or weight == {}:
                # None or {} is updated to be 1
                graph.update_edge_by_index(edge_index, 1)
            elif not isinstance(weight, numbers.Number):
                raise ValueError(f"Unsupported weight {weight} on edge with index {edge_index}.")

        self._graph = graph

        self.pos: Optional[dict] = None

    @property
    def graph(self) -> PyGraph:
        """Return a copy of the input graph."""
        return self._graph.copy()

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return self.graph.num_nodes()

    @property
    def node_indexes(self) -> NodeIndices:
        """Return the node indexes."""
        return self.graph.node_indexes()

    @property
    def weighted_edge_list(self) -> WeightedEdgeList:
        """Return a list of weighted edges."""
        return self.graph.weighted_edge_list()

    def copy(self) -> "Lattice":
        """Return a copy of the lattice."""
        return deepcopy(self)

    @classmethod
    def from_nodes_and_edges(
        cls, num_nodes: int, weighted_edges: List[Tuple[int, int, complex]]
    ) -> "Lattice":
        """Return an instance of Lattice from the number of nodes and the list of edges.

        Args:
            num_nodes: The number of nodes.
            weighted_edges: A list of tuples consisting of two nodes and the weight between them.
        Returns:
            Lattice generated from lists of nodes and edges.
        """
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(weighted_edges)
        return cls(graph)

    def to_adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        """Return its adjacency matrix from weighted edges.
        The weighted edge list is interpreted as the upper triangular matrix.
        Defaults to False.

        Args:
            weighted: The matrix elements are 0 or 1 when it is False.
                Otherwise, the weights on edges are returned as the matrix elements.

        Returns:
            The adjacency matrix of the input graph.
        """
        if weighted:
            real_part = adjacency_matrix(self.graph, weight_fn=np.real)
            imag_part = adjacency_matrix(self.graph, weight_fn=np.imag)
            imag_part = np.triu(imag_part) - np.triu(imag_part).T
            ad_mat = real_part + 1.0j * imag_part

        else:
            ad_mat = adjacency_matrix(self.graph, weight_fn=lambda x: 1)

        return ad_mat

    @staticmethod
    def _mpl(graph: PyGraph, self_loop: bool, **kwargs):
        """
        Auxiliary function for drawing the lattice using matplotlib.

        Args:
            graph : graph to be drawn.
            self_loop : Draw self-loops, which are edges connecting a node to itself.
            **kwargs : Kwargs for drawing the lattice.

        Raises:
            MissingOptionalLibraryError: Requires matplotlib.
        """
        if not HAS_MATPLOTLIB:
            raise MissingOptionalLibraryError(
                libname="Matplotlib", name="_mpl", pip_install="pip install matplotlib"
            )
        from matplotlib import pyplot as plt

        if not self_loop:
            self_loops = [(i, i) for i in range(graph.num_nodes()) if graph.has_edge(i, i)]
            graph.remove_edges_from(self_loops)

        mpl_draw(
            graph=graph,
            **kwargs,
        )
        plt.draw()

    def draw(
        self,
        self_loop: bool = False,
        style: Optional[LatticeDrawStyle] = None,
    ):
        """Draw the lattice.

        Args:
            self_loop : Draw self-loops in the lattice. Defaults to False.
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
            style.pos = self.pos

        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **asdict(style),
        )
