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

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

import qiskit_nature.results.protein_folding_result as pfr


class ProteinPlotter:
    """This class is a plotter class for the ProteinFoldingResult"""

    def __init__(self, protein_folding_result: pfr.ProteinFoldingResult) -> None:
        """
        Args:
            protein_folding_result: The protein folding result to be plotted
        """

        self._protein_folding_result = protein_folding_result
        
        self.x_main,self.y_main,self.z_main = np.split(self._protein_folding_result.protein_shape_file_gen.main_positions.transpose(),3,0)
        self.x_main,self.y_main,self.z_main =self.x_main[0], self.y_main[0], self.z_main[0]
        
        
        self.fig = plt.figure()
        self.ax_graph = self.fig.add_subplot(projection="3d")

        
        

    def draw_main_chain(self):
        """
        Draws the main chain on a subplot object.

        """        
        for i, main_aminoacid in enumerate(
            self._protein_folding_result.protein_shape_file_gen._main_chain_aminoacid_list):
            self.ax_graph.text(self.x_main[i], self.y_main[i], self.z_main[i], main_aminoacid, size=10, zorder=10, color="k")
            
        self.ax_graph.plot3D(self.x_main, self.y_main, self.z_main)
        return self.ax_graph.scatter3D(self.x_main, self.y_main, self.z_main, s=500, label="Main Chain")



    def draw_side_chains(self):
        """
        Draws the side chain on a subplot object.

        """
        side_positions = self._protein_folding_result.protein_shape_file_gen.side_positions
        side_aminoacids = self._protein_folding_result.protein_shape_file_gen._main_chain_aminoacid_list
        for i, side_chain in enumerate(side_positions):
            if side_chain is not None:
                x_side, y_side, z_side = side_chain
                side_scatter = self.ax_graph.scatter3D(
                    x_side, y_side, z_side, s=600, c="green", label="Side Chain"
                )
                self.ax_graph.plot3D([self.x_main[i], x_side], [self.y_main[i], y_side], [self.z_main[i], z_side], c="green")
                self.ax_graph.text(
                                    x_side,
                                    y_side,
                                    z_side,
                                    side_aminoacids[i],
                                    size=10,
                                    zorder=10,
                                    color="k",
                                )
        return side_scatter
    
    def format_graph(self,title,ticks,grid):
        """
        Formats the plot.
        Args:
            title: The title of the plot.
            ticks: Boolean for showing ticks in the graphic.
            grid: Boolean for showing the grid in the graphic.
            ax_graph: Subplot that we want to edit.
            fig: Figure that we want to modify.
            main_scatter: Scattering object that we will use for the legend.
            side_scatter: Scattering object that we will use for the legend.
        """
                
        self.ax_graph.set_box_aspect([1, 1, 1])

        self.ax_graph.grid(grid)
        
        if not ticks:
            self.ax_graph.set_xticks([])
            self.ax_graph.set_yticks([])
            self.ax_graph.set_zticks([])

        self.ax_graph.set_xlabel("x")        
        self.ax_graph.set_ylabel("y")
        self.ax_graph.set_zlabel("z")

        self.fig.legend(handles=[self.main_scatter, self.side_scatter], labelspacing=2,markerscale = 0.5)
        self.ax_graph.set_title(title)
    
    def plot(
        self, title: str = "Protein Structure", ticks: bool = False, grid: bool = False
    ) -> None:
        """
        Plots the molecule in 3D.
        Args:
            title: The title of the plot.
            ticks: Boolean for showing ticks in the graphic.
            grid: Boolean for showing the grid in the graphic.
        """        
        

        self.main_scatter = self.draw_main_chain()
        self.side_scatter = self.draw_side_chains()
        
        self.format_graph(title=title,
                          ticks = ticks,
                          grid=grid)
        

        plt.draw()









































