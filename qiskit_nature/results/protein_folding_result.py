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
"""The protein folding result."""
from typing import List, Optional
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.results import EigenstateResult
from .utils.protein_shape_decoder import ProteinShapeDecoder
from .utils.protein_shape_file_gen import ProteinShapeFileGen
from .utils.protein_plotter import ProteinPlotter


class ProteinFoldingResult(EigenstateResult):
    """
    The Protein Folding Result.
    """

    def __init__(
        self,
        peptide: Peptide,
        unused_qubits: List[int],
        turns_sequence: str,
    ) -> None:
        """
        Args:
            peptide: The peptide defining the protein subject to the folding problem.
            unused_qubits: The list of indices for qubits in the original problem formulation that were
            removed during compression.
            turns_sequence: The bit sequence encoding the turns of the shape of the protein.

        """
        super().__init__()

        self._turns_sequence = turns_sequence
        self._unused_qubits = unused_qubits
        self._peptide = peptide
        self._main_chain_length = len(self._peptide.get_main_chain.main_chain_residue_sequence)
        self._side_chain_hot_vector = self._peptide.get_side_chain_hot_vector()

        self._protein_shape_decoder = ProteinShapeDecoder(
            turns_sequence=self._turns_sequence,
            side_chain_hot_vector=self._side_chain_hot_vector,
            fifth_bit=5 in self._unused_qubits[:6],
        )

        self._protein_shape_file_gen = ProteinShapeFileGen(
            self.protein_shape_decoder.main_turns,
            self.protein_shape_decoder.side_turns,
            self._peptide,
        )

    @property
    def protein_shape_decoder(self) -> ProteinShapeDecoder:
        """Returns the :class:`ProteinShapeDecoder` of the result.
        This class will interpret the result bitstring and return the encoded information."""
        return self._protein_shape_decoder

    @property
    def protein_shape_file_gen(self) -> ProteinShapeFileGen:
        """Returns the :class:`ProteinShapeFileGen` of the result."""
        return self._protein_shape_file_gen

    @property
    def turns_sequence(self) -> str:
        """Returns the best sequence."""
        return self._turns_sequence

    def get_result_binary_vector(self) -> str:
        """Returns a string that encodes a solution of the :class:`ProteinFoldingProblem`.
        The :class:`ProteinFoldingProblem` uses a compressed optimization problem that does not match the
        number of qubits in the original objective function. This method calculates the original
        version of the solution vector. Bits that can take any value without changing the
        solution are denoted by '*'."""
        unused_qubits = self._unused_qubits
        result = []
        offset = 0
        size = len(self._turns_sequence)
        for i in range(size):
            index = size - 1 - i
            while i + offset in unused_qubits:
                result.append("*")
                offset += 1
            result.append(self._turns_sequence[index])

        return "".join(result[::-1])

    def save_xyz_file(self, name: Optional[str] = None, path: str = "") -> None:
        """
        Generates a .xyz file.
        Args:
            name: Name of the file to be generated. If the name is ``None`` the
            name of the file will be the letters of the aminoacids on the main_chain.
            path: Path where the file will be generated. If left empty the file will
            be saved in the working directory.
        """
        if name is None:
            name = str(self._peptide.get_main_chain.main_chain_residue_sequence)
        self.protein_shape_file_gen.save_xyz_file(name=name, path=path)

    def plot_folded_protein(
        self, title: str = "Protein Structure", ticks: bool = True, grid: bool = False
    ) -> None:
        """
        Plots the molecule in 3D.
        Args:
            title: The title of the plot.
            ticks: Boolean for showing ticks in the graphic.
            grid: Boolean for showing the grid in the graphic.

        """
        ProteinPlotter(self.protein_shape_file_gen).plot(title=title, ticks=ticks, grid=grid)
