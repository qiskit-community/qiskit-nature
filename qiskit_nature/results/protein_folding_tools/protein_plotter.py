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
"""An auxiliary class that plots the aminoacids of a molecule
 in ProteinFoldingResult """

from __future__ import annotations
import matplotlib.pyplot as plt

import qiskit_nature.results.protein_folding_result as pfr


class ProteinPlotter:
    """This class is a plotter class for the ProteinFoldingResult"""

    def __init__(self, protein_folding_result: pfr.ProteinFoldingResult) -> None:
        """
        Args:
            protein_folding_result: The protein folding result to be plotted
        """

        self._protein_folding_result = protein_folding_result

    def plot(
        self, title: str = "Protein Structure", ticks: bool = False, grid: bool = False
    ) -> None:
        """
        Plots the molecule in 3D.
        Args:
            title: The title of the plot
            ticks: Boolean for showing ticks in the graphic
            grid: Boolean for showing the grid in the graphic
        """
        main_positions = self._protein_folding_result.protein_xyz.main_positions
        side_positions = self._protein_folding_result.protein_xyz.side_positions

        fig = plt.figure()
        ax_graph = fig.add_subplot(projection="3d")

        x, y, z = main_positions[:, 0], main_positions[:, 1], main_positions[:, 2]
        for i, main_aminoacid in enumerate(
            self._protein_folding_result.protein_xyz._main_chain_aminoacid_list
        ):
            ax_graph.text(x[i], y[i], z[i], main_aminoacid, size=10, zorder=10, color="k")
        ax_graph.plot3D(x, y, z)
        main_scatter = ax_graph.scatter3D(x, y, z, s=500, label="Main Chain")

        side_aminoacids = self._protein_folding_result.protein_xyz._main_chain_aminoacid_list
        for i, side_chain in enumerate(side_positions):
            if side_chain is not None:
                x_side, y_side, z_side = side_chain
                side_scatter = ax_graph.scatter3D(
                    x_side, y_side, z_side, s=600, c="green", label="Side Chain"
                )
                ax_graph.plot3D([x[i], x_side], [y[i], y_side], [z[i], z_side], c="green")
                ax_graph.text(
                    x_side,
                    y_side,
                    z_side,
                    side_aminoacids[i],
                    size=10,
                    zorder=10,
                    color="k",
                )
                
        ax_graph.set_box_aspect([1, 1, 1])

        ax_graph.grid(grid)
        
        if not ticks:
            ax_graph.set_xticks([])
            ax_graph.set_yticks([])
            ax_graph.set_zticks([])

        ax_graph.set_xlabel("x")        
        ax_graph.set_ylabel("y")
        ax_graph.set_zlabel("z")

        fig.legend(handles=[main_scatter, side_scatter], labelspacing=2,markerscale = 0.5)
        ax_graph.set_title(title)

        plt.draw()
