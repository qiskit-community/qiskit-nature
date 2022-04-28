# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:13:19 2022

@author: 7G5381848
"""
import matplotlib.pyplot as plt
import numpy as np


class ProteinPlotter:
    """This class is a plotter class for the ProteinFoldingResult"""

    def __init__(self, proteinfoldingresult) -> None:
        self._proteinfoldingresult = proteinfoldingresult

    def plot(self) -> None:
        """
        Plots the molecule in 3D.
        """
        main_positions = self._proteinfoldingresult.protein_xyz.main_positions
        side_positions = self._proteinfoldingresult.protein_xyz.side_positions

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        X, Y, Z = main_positions[:, 0], main_positions[:, 1], main_positions[:, 2]
        ax.plot3D(X, Y, Z)
        ax.scatter3D(X, Y, Z, s=500)

        for i, side_chain in enumerate(side_positions):
            if type(side_chain) == np.ndarray:
                Xs, Ys, Zs = side_chain
                ax.scatter3D(Xs, Ys, Zs, s=600, c="green")
                ax.plot3D([X[i], Xs], [Y[i], Ys], [Z[i], Zs], c="green")

        ax.set_box_aspect([1,1,1])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        plt.draw()
