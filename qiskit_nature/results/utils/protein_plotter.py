# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An auxiliary class that plots aminoacids of a molecule
 in a ProteinFoldingResult."""
import numpy as np
from qiskit.utils import optionals as _optionals

from qiskit_nature.results.utils.protein_shape_file_gen import ProteinShapeFileGen

if _optionals.HAS_MATPLOTLIB:
    # pylint: disable=import-error,unused-import
    from matplotlib.pyplot import figure
    from matplotlib.axes import Axes


@_optionals.HAS_MATPLOTLIB.require_in_instance
class ProteinPlotter:
    """Plotter class for ProteinFoldingResult."""

    def __init__(self, shape_gen: ProteinShapeFileGen) -> None:
        """
        Args:
            shape_gen: :class:`ProteinShapeFileGen` with the shape to be plotted.
        """
        # pylint: disable=import-error
        import matplotlib.pyplot as plt

        self._shape_gen = shape_gen
        # pylint: disable=unbalanced-tuple-unpacking
        (
            self._x_main,
            self._y_main,
            self._z_main,
        ) = np.split(self._shape_gen.main_positions.transpose(), 3, 0)
        self._x_main, self._y_main, self._z_main = self._x_main[0], self._y_main[0], self._z_main[0]

        self._fig = plt.figure()
        self._ax_graph = self._fig.add_subplot(projection="3d")

    def _draw_main_chain(self):
        """
        Draws the main chain.

        """
        for i, main_aminoacid in enumerate(self._shape_gen._main_chain_aminoacid_list):
            self._ax_graph.text(
                self._x_main[i],
                self._y_main[i],
                self._z_main[i],
                main_aminoacid,
                size=10,
                zorder=10,
                color="k",
            )

        self._ax_graph.plot3D(self._x_main, self._y_main, self._z_main)
        return self._ax_graph.scatter3D(
            self._x_main, self._y_main, self._z_main, s=500, label="Main Chain"
        )

    def _draw_side_chains(self):
        """
        Draws the side chain.
        """
        side_positions = self._shape_gen.side_positions
        side_aminoacids = self._shape_gen._main_chain_aminoacid_list
        for i, side_chain in enumerate(side_positions):
            if side_chain is not None:
                x_side, y_side, z_side = side_chain
                side_scatter = self._ax_graph.scatter3D(
                    x_side, y_side, z_side, s=600, c="green", label="Side Chain"
                )
                self._ax_graph.plot3D(
                    [self._x_main[i], x_side],
                    [self._y_main[i], y_side],
                    [self._z_main[i], z_side],
                    c="green",
                )
                self._ax_graph.text(
                    x_side,
                    y_side,
                    z_side,
                    side_aminoacids[i],
                    size=10,
                    zorder=10,
                    color="k",
                )
        return side_scatter

    def _format_graph(
        self, title: str, ticks: bool, grid: bool, main_scatter: "Axes", side_scatter: "Axes"
    ):
        """
        Formats the plot.
        Args:
            title: The title of the plot.
            ticks: Boolean for showing ticks in the graphic.
            grid: Boolean for showing the grid in the graphic.
            main_scatter: Scattering object that we will use for the legend.
            side_scatter: Scattering object that we will use for the legend.
        """

        self._ax_graph.set_box_aspect([1, 1, 1])

        self._ax_graph.grid(grid)

        if not ticks:
            self._ax_graph.set_xticks([])
            self._ax_graph.set_yticks([])
            self._ax_graph.set_zticks([])

        self._ax_graph.set_xlabel("x")
        self._ax_graph.set_ylabel("y")
        self._ax_graph.set_zlabel("z")

        self._fig.legend(handles=[main_scatter, side_scatter], labelspacing=2, markerscale=0.5)
        self._ax_graph.set_title(title)

    def get_figure(
        self, title: str = "Protein Structure", ticks: bool = False, grid: bool = False
    ) -> "figure":
        """
        Plots the molecule in 3D.

        Args:
            title: The title of the plot.
            ticks: Boolean for showing ticks in the graphic.
            grid: Boolean for showing the grid in the graphic.
        Returns:
            A figure with the folded protein.
        """

        main_scatter = self._draw_main_chain()
        side_scatter = self._draw_side_chains()

        self._format_graph(
            title=title,
            ticks=ticks,
            grid=grid,
            main_scatter=main_scatter,
            side_scatter=side_scatter,
        )

        return self._fig
