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

"""General Lattice."""
from functools import wraps
from inspect import signature
from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from retworkx import NodeIndices, PyGraph, WeightedEdgeList, adjacency_matrix
from retworkx.visualization import mpl_draw


def _add_signature(func_with_signature):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        sig = signature(func)
        sig_func = signature(func_with_signature)
        post_params = []
        pre_params = []

        for k, v in sig.parameters.items():
            if str(v)[0] == "*":
                post_params.append(sig.parameters[k])
            else:
                pre_params.append(sig.parameters[k])

        wrapper.__signature__ = sig.replace(
            parameters=pre_params + list(sig_func.parameters.values()) + post_params
        )
        wrapper.__annotations__ = {**func.__annotations__, **func_with_signature.__annotations__}

        return wrapper

    return decorator


# pylint: disable=unused-argument
def _draw_signature(
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
):
    pass


_add_draw_signature = _add_signature(_draw_signature)


class Lattice:
    """General Lattice."""

    def __init__(self, graph: PyGraph) -> None:
        """
        Args:
            graph: Input graph for Lattice. `graph.multigraph` must be False.

        Raises:
            ValueError: If `graph.multigraph` is True for a given graph, it is invalid.
        """
        if graph.multigraph:
            raise ValueError(
                f"Invalid `graph.multigraph` {graph.multigraph} is given. "
                "`graph.multigraph` must be `False`."
            )
        if graph.edges() == [None] * graph.num_edges():
            weighted_edges = [edge + (1.0,) for edge in graph.edge_list()]
            for start, end, weight in weighted_edges:
                graph.update_edge(start, end, weight)
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
        return Lattice(self.graph.copy())

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

    def to_adjacency_matrix(self, weighted: bool = True) -> np.ndarray:
        """Return its adjacency matrix from weighted edges.
        The weighted edge list is interpreted as the upper triangular matrix.
        Defaults to True.

        Args:
            weighted: The matrix elements are 0 or 1 when it is true.
                Otherwise, the weights on edges are returned as a matrix.

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

        Returns:
            A matplotlib figure for the visualization if not running with an
            interactive backend (like in jupyter) or if ``ax`` is not set.
        """

        if not self_loop:
            self_loops = [(i, i) for i in range(graph.num_nodes()) if graph.has_edge(i, i)]
            graph.remove_edges_from(self_loops)

        mpl_draw(
            graph=graph,
            **kwargs,
        )
        plt.draw()

    @_add_draw_signature
    def draw(
        self,
        self_loop: bool = False,
        **kwargs,
    ):
        """Draw the lattice.

        Args:
            self_loop : Draw self-loops in the lattice. Defaults to False.
            **kwargs : Kwargs for retworkx.visualization.mpl_draw.
                Please see
                https://qiskit.org/documentation/retworkx/stubs/retworkx.visualization.mpl_draw.html#retworkx.visualization.mpl_draw
                for details.
        """

        graph = self.graph

        if "pos" not in kwargs:
            kwargs["pos"] = self.pos
        self._mpl(
            graph=graph,
            self_loop=self_loop,
            **kwargs,
        )
