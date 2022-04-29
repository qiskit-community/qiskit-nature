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

    def plot(self,ticks:bool = True,grid:bool = False) -> None:
        """
        Plots the molecule in 3D.
        """
        main_positions = self._protein_folding_result.protein_xyz.main_positions
        side_positions = self._protein_folding_result.protein_xyz.side_positions

        
        fig = plt.figure()
        ax_graph = fig.add_subplot(projection="3d")

        x, y, z = main_positions[:, 0], main_positions[:, 1], main_positions[:, 2]
        for i ,main_aminoacid in enumerate(self._protein_folding_result.protein_xyz._main_chain_aminoacid_list):
            ax_graph.text(x[i],y[i],z[i],  '%s' % (main_aminoacid), size=20, zorder=10,  color='k')
        ax_graph.plot3D(x, y, z)
        ax_graph.scatter3D(x, y, z, s=500)

        side_aminoacids = self._protein_folding_result.protein_xyz._main_chain_aminoacid_list
        for i, side_chain in enumerate(side_positions):
            if side_chain is not None:
                x_side, y_side, z_side = side_chain
                ax_graph.scatter3D(x_side, y_side, z_side, s=600, c="green")
                ax_graph.plot3D([x[i], x_side], [y[i], y_side], [z[i], z_side], c="green")
                ax_graph.text(x_side,y_side,z_side,  '%s' % (side_aminoacids[i]), size=20, zorder=10,color='k')

        ax_graph.set_box_aspect([1, 1, 1])
        
        if grid:
            ax_graph.grid(False)
        if ticks:
            ax_graph.set_xticks([])
            ax_graph.set_yticks([])
            ax_graph.set_zticks([])        
        
        ax_graph.set_xlabel('x')
        ax_graph.set_ylabel('y')
        ax_graph.set_zlabel('z')
        


        plt.draw()
