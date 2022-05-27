# This code is part of Qiskit.
#
# (C) Copyright IBM 2021,2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The protein folding result."""
from __future__ import annotations


import qiskit_nature.problems.sampling.protein_folding.protein_folding_problem as pfp

from qiskit_nature.results import EigenstateResult
from .utils.protein_decoder import ProteinDecoder
from .utils.protein_shape_file_gen import ProteinShapeFileGen
from .utils.protein_plotter import ProteinPlotter


class ProteinFoldingResult(EigenstateResult):
    """
    The Protein Folding Result.
    """

    def __init__(
        self,
        protein_folding_problem: pfp.ProteinFoldingProblem,
        best_sequence: str,
    ) -> None:
        """
        Args:
            protein_folding_problem: The protein folding problem that led to the result.
            best_sequence: The best sequence in the result eigenstate.

        """
        super().__init__()
        self._protein_folding_problem = protein_folding_problem
        self._best_sequence = best_sequence
        self._unused_qubits = self._protein_folding_problem.unused_qubits
        self._main_chain_length = len(
            self._protein_folding_problem.peptide.get_main_chain.main_chain_residue_sequence
        )
        self._side_chain_hot_vector = (
            self._protein_folding_problem.peptide.get_side_chain_hot_vector()
        )

        self._protein_decoder = ProteinDecoder(
            best_sequence=self._best_sequence,
            side_chain_hot_vector=self._side_chain_hot_vector,
            fifth_bit=5 in self._unused_qubits[:6],
        )

        self._protein_shape_file_gen = ProteinShapeFileGen(
            self.protein_decoder.main_turns,
            self.protein_decoder.side_turns,
            self._protein_folding_problem.peptide,
        )

    @property
    def protein_decoder(self) -> ProteinDecoder:
        """Returns the ProteinDecoder of the result.
        This class will interpret the result bitstring and return the encoded information."""
        return self._protein_decoder

    @property
    def protein_shape_file_gen(self) -> ProteinShapeFileGen:
        """Returns the ProteinShapeFileGen of the result."""
        return self._protein_shape_file_gen

    @property
    def best_sequence(self) -> str:
        """Returns the best sequence."""
        return self._best_sequence

    def get_result_binary_vector(self) -> str:
        """Returns a string that encodes a solution of the ProteinFoldingProblem.
        The ProteinFoldingProblem uses a compressed optimization problem that does not match the
        number of qubits in the original objective function. This method calculates the original
        version of the solution vector. Bits that can take any value without changing the
        solution are denoted by '*'."""
        unused_qubits = self._unused_qubits
        result = []
        offset = 0
        size = len(self._best_sequence)
        for i in range(size):
            index = size - 1 - i
            while i + offset in unused_qubits:
                result.append("*")
                offset += 1
            result.append(self._best_sequence[index])

        return "".join(result[::-1])

    def save_xyz_file(self, name: str = None, path: str = "") -> None:
        """
        Generates a .xyz file.
        Args:
            name: Name of the file to be generated.
            path: Path where the file will be generated.
        """
        if name is None:
            name = str(
                self._protein_folding_problem.peptide.get_main_chain.main_chain_residue_sequence
            )
        self.protein_shape_file_gen.save_xyz_file(name=name, path=path)

    def plot_folded_protein(
        self, title: str = "Protein Structure", ticks: bool = True, grid: bool = False
    ) -> None:
        """
        Plots the molecule in 3D.
        Args:
            title: The title of the plot
            ticks: Boolean for showing ticks in the graphic
            grid: Boolean for showing the grid in the graphic

        """
        ProteinPlotter(self).plot(title=title, ticks=ticks, grid=grid)
