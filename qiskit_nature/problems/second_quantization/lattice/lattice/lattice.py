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

"""The Lattice class"""
from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from retworkx import NodeIndices, PyGraph, WeightedEdgeList, adjacency_matrix
from retworkx.visualization import mpl_draw


class Lattice:
    """Lattice class."""

    def __init__(self, graph: PyGraph) -> None:
        """
        Args:
            graph: Input graph for Lattice. `multigraph` must be False.

        Raises:
            ValueError: A given graph is invalid.
        """
        if graph.multigraph:
            raise ValueError(
                f"Invalid `multigraph` {graph.multigraph} is given. `multigraph` must be `False`."
            )
        if graph.edges() == [None] * graph.num_edges():
            weighted_edges = [edge + (1.0,) for edge in graph.edge_list()]
            for start, end, weight in weighted_edges:
                graph.update_edge(start, end, weight)
        self._graph = graph

    @property
    def graph(self) -> PyGraph:
        """Return a copy of the input graph."""
        return self._graph.copy()

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return self.graph.num_nodes()

    @property
    def nodes(self) -> NodeIndices:
        """Return the node indexes."""
        return self.graph.node_indexes()

    @property
    def weighted_edge_list(self) -> WeightedEdgeList:
        """Return a list of weighted edges."""
        return self.graph.weighted_edge_list()

    def copy(self) -> "Lattice":
        """Return a copy of the lattice."""
        return Lattice(self.graph.copy())

    @classmethod
    def from_adjacency_matrix(cls, input_adjacency_matrix: np.ndarray) -> "Lattice":
        """Return an instance of Lattice from a given hopping_matrix.

        Args:
            input_adjacency_matrix: Adjacency matrix with real or complex matrix elements.

        Returns:
            Lattice generated from a given adjacency_matrix.
        """

        col_length, row_length = input_adjacency_matrix.shape
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(col_length))
        for source_index in range(col_length):
            for target_index in range(source_index, row_length):
                weight = input_adjacency_matrix[source_index, target_index]
                if not weight == 0.0:
                    graph.add_edge(source_index, target_index, weight)

        return cls(graph)

    @classmethod
    def from_nodes_edges(
        cls, num_nodes: int, weighted_edges: List[Tuple[int, int, complex]]
    ) -> "Lattice":
        """Return an instance of Lattice from the number of nodes and the list of edges.

        Returns:
            num_nodes: The number of nodes.
            weighted_edges: A list of tuples consisting of two nodes and the weight between them.
        """
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(weighted_edges)
        return cls(graph)

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return the hopping matrix from weighted edges.
        The weighted edge list is interpreted as the upper triangular matrix.
        """
        real_part = adjacency_matrix(self.graph, weight_fn=np.real)
        real_part = real_part - (1 / 2) * np.diag(real_part.diagonal())
        imag_part = adjacency_matrix(self.graph, weight_fn=np.imag)
        imag_part = np.triu(imag_part) - np.triu(imag_part).T
        return real_part + 1.0j * imag_part

    def draw(
        self,
        pos: Optional[dict] = None,
        ax: Optional[Axes] = None,
        arrows: bool = True,
        arrowstyle: Optional[str] = None,
        arrow_size: int = 10,
        with_labels: bool = False,
        node_list: Optional[list] = None,
        edge_list: Optional[list] = None,
        node_size: Union[int, list] = 300,
        node_color: Union[
            str,
            Tuple[float, float, float],
            Tuple[float, float, float],
            List[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]],
        ] = "#1f78b4",
        node_shape: str = "o",
        alpha: Optional[float] = None,
        cmap: Optional[Colormap] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        linewidths: Union[float, Sequence] = 1.0,
        width: Union[float, Sequence] = 1.0,
        edge_color: Union[str, Sequence] = "k",
        edge_cmap: Optional[Colormap] = None,
        edge_vmin: Optional[float] = None,
        edge_vmax: Optional[float] = None,
        style: str = "solid",
        labels: Optional[Callable] = None,
        edge_labels: Optional[Callable] = None,
        font_size: int = 12,
        font_color: str = "k",
        font_weight: str = "normal",
        font_family: str = "sans-serif",
        label: Optional[str] = None,
        connectionstyle: str = "arc3",
        self_loop: bool = False,
        **kwargs,
    ):
        r"""Draw the lattice.

        Args:
            pos: An optional dictionary (or
                a :class:`~retworkx.Pos2DMapping` object) with nodes as keys and
                positions as values. If not specified a spring layout positioning will
                be computed. See `layout_functions` for functions that compute
                node positions.
            ax: An optional Matplotlib Axes object to draw the
                graph in.
            arrows: For :class:`~retworkx.PyDiGraph` objects if ``True``
                draw arrowheads. (defaults to ``True``) Note, that the Arrows will
                be the same color as edges.
            arrowstyle: An optional string for directed graphs to choose
                the style of the arrowsheads. See
                :class:`matplotlib.patches.ArrowStyle` for more options. By default the
                value is set to ``'-\|>'``.
            arrow_size: For directed graphs, choose the size of the arrow
                head's length and width. See
                :class:`matplotlib.patches.FancyArrowPatch` attribute and constructor
                kwarg ``mutation_scale`` for more info. Defaults to 10.
            with_labels: Set to ``True`` to draw labels on the nodes. Edge
                labels will only be drawn if the ``edge_labels`` parameter is set to a
                function. Defaults to ``False``.

            node_list: An optional list of node indices in the graph to
                draw. If not specified all nodes will be drawn.
            edge_list: An option list of edges in the graph to draw. If not
                specified all edges will be drawn
            node_size: Optional size of nodes. If an array is
                specified it must be the same length as node_list. Defaults to 300
            node_color: Optional node color. Can be a single color or
                a sequence of colors with the same length as node_list. Color can be
                string or rgb (or rgba) tuple of floats from 0-1. If numeric values
                are specified they will be mapped to colors using the ``cmap`` and
                ``vmin``,``vmax`` parameters. See :func:`matplotlib.scatter` for more
                details. Defaults to ``'#1f78b4'``)
            node_shape: The optional shape node. The specification is the
                same as the :func:`matplotlib.pyplot.scatter` function's ``marker``
                kwarg, valid options are one of
                ``['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8']``. Defaults to
                ``'o'``
            alpha: Optional value for node and edge transparency
            cmap: An optional Matplotlib colormap
                object for mapping intensities of nodes
            vmin: Optional minimum value for node colormap scaling
            vmax: Optional minimum value for node colormap scaling
            linewidths: An optional line width for symbol
                borders. If a sequence is specified it must be the same length as
                node_list. Defaults to 1.0
            width: An optional width to use for edges. Can
                either be a float or sequence  of floats. If a sequence is specified
                it must be the same length as node_list. Defaults to 1.0
            edge_color: color or array of colors (default='k')
                Edge color. Can be a single color or a sequence of colors with the same
                length as edge_list. Color can be string or rgb (or rgba) tuple of
                floats from 0-1. If numeric values are specified they will be
                mapped to colors using the ``edge_cmap`` and ``edge_vmin``,
                ``edge_vmax`` parameters.
            edge_cmap: An optional Matplotlib
                colormap for mapping intensities of edges.
            edge_vmin: Optional minimum value for edge colormap scaling
            edge_vmax: Optional maximum value for node colormap scaling
            style: An optional string to specify the edge line style.
                For example, ``'-'``, ``'--'``, ``'-.'``, ``':'`` or words like
                ``'solid'`` or ``'dashed'``. See the
                :class:`matplotlib.patches.FancyArrowPatch` attribute and kwarg
                ``linestyle`` for more details. Defaults to ``'solid'``.
            labels: An optional callback function that will be passed a
                node payload and return a string label for the node. For example::

                    labels=str

                could be used to just return a string cast of the node's data payload.
                Or something like::

                    labels=lambda node: node['label']

                could be used if the node payloads are dictionaries.
            edge_labels: An optional callback function that will be passed
                an edge payload and return a string label for the edge. For example::

                    edge_labels=str

                could be used to just return a string cast of the edge's data payload.
                Or something like::

                    edge_labels=lambda edge: edge['label']

                could be used if the edge payloads are dictionaries. If this is set
                edge labels will be drawn in the visualization.
            font_size: An optional fontsize to use for text labels, By
                default a value of 12 is used for nodes and 10 for edges.
            font_color: An optional font color for strings. By default
                ``'k'`` (ie black) is set.
            font_weight: An optional string used to specify the font weight.
                By default a value of ``'normal'`` is used.
            font_family: An optional font family to use for strings. By
                default ``'sans-serif'`` is used.
            label: An optional string label to use for the graph legend.
            connectionstyle: An optional value used to create a curved arc
                of rounding radius rad. For example,
                ``connectionstyle='arc3,rad=0.2'``. See
                :class:`matplotlib.patches.ConnectionStyle` and
                :class:`matplotlib.patches.FancyArrowPatch` for more info. By default
                this is set to ``"arc3"``.
            self_loop: Draw self-loops in a lattice. Defaults to False.
            **kwargs: Kwargs for drawing the lattice.

        Returns:
            A matplotlib figure for the visualization if not running with an
            interactive backend (like in jupyter) or if ``ax`` is not set.
        """
        if self_loop:
            mpl_draw(
                graph=self.graph,
                pos=pos,
                ax=ax,
                arrows=arrows,
                arrowstyle=arrowstyle,
                arrow_size=arrow_size,
                with_labels=with_labels,
                node_list=node_list,
                edge_list=edge_list,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=linewidths,
                width=width,
                edge_color=edge_color,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                style=style,
                labels=labels,
                edge_labels=edge_labels,
                font_size=font_size,
                font_color=font_color,
                font_weight=font_weight,
                font_family=font_family,
                label=label,
                connectionstyle=connectionstyle,
                **kwargs,
            )
            plt.draw()
        elif not self_loop:
            graph_no_loop = self.graph
            self_loops = [(i, i) for i in range(self.num_nodes) if graph_no_loop.has_edge(i, i)]
            graph_no_loop.remove_edges_from(self_loops)
            mpl_draw(
                graph=graph_no_loop,
                pos=pos,
                ax=ax,
                arrows=arrows,
                arrowstyle=arrowstyle,
                arrow_size=arrow_size,
                with_labels=with_labels,
                node_list=node_list,
                edge_list=edge_list,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=linewidths,
                width=width,
                edge_color=edge_color,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                style=style,
                labels=labels,
                edge_labels=edge_labels,
                font_size=font_size,
                font_color=font_color,
                font_weight=font_weight,
                font_family=font_family,
                label=label,
                connectionstyle=connectionstyle,
                **kwargs,
            )
            plt.draw()
