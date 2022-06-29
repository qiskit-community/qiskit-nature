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
from qiskit.utils import optionals as _optionals
from qiskit_nature.second_quantization.problems import EigenstateResult

from ..problems.sampling.protein_folding.peptide.peptide import Peptide
from .utils.protein_plotter import ProteinPlotter
from .utils.protein_shape_decoder import ProteinShapeDecoder
from .utils.protein_shape_file_gen import ProteinShapeFileGen


if _optionals.HAS_MATPLOTLIB:
    # pylint: disable=import-error,unused-import
    from matplotlib.pyplot import figure


class ProteinFoldingResult(EigenstateResult):
    """
    The Protein Folding Result.
    This class interprets a bitstring encoding the turns of a protein from
    :class:`~qiskit_nature.problems.sampling.protein_folding_problem.ProteinFoldingProblem`
    and decodes it. One can generate a .xyz file
    (using :meth:`~qiskit_nature.results.ProteinFoldingResult.save_xyz_file`), which is a file
    containing the cartesian coordinates of each atom in the protein. This kind of file can be
    used with other software to generate plots of the molecule.
    Alternatively, one can use :meth:`~qiskit_nature.results.ProteinFoldingResult.get_figure`.
    Note that `matplotlib` needs to be installed in order to generate such a figure.
    """

    def __init__(
        self,
        peptide: Peptide,
        unused_qubits: List[int],
        turn_sequence: str,
    ) -> None:
        """
        Args:
            peptide: The peptide defining the protein subject to the folding problem.
            unused_qubits: The list of indices for qubits in the original problem formulation that were
                removed during compression.
            turn_sequence: The bit sequence encoding the turns of the shape of the protein.

        """
        super().__init__()

        self._turn_sequence = turn_sequence
        self._unused_qubits = unused_qubits
        self._peptide = peptide
        self._main_chain_length = len(self._peptide.get_main_chain.main_chain_residue_sequence)
        self._side_chain_hot_vector = self._peptide.get_side_chain_hot_vector()

        self._protein_shape_decoder = ProteinShapeDecoder(
            turn_sequence=self._turn_sequence,
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
    def turn_sequence(self) -> str:
        """Returns the best sequence."""
        return self._turn_sequence

    def get_result_binary_vector(self) -> str:
        """Returns a string that encodes a solution of the
        :class:`~qiskit_nature.problems.sampling.protein_folding_problem.ProteinFoldingProblem`.
        The :class:`~qiskit_nature.problems.sampling.protein_folding_problem.ProteinFoldingProblem`
        uses a compressed optimization problem that does not match the
        number of qubits in the original objective function. This method calculates the original
        version of the solution vector. Bits that can take any value without changing the
        solution are denoted by '_'.
        This string is read from right to left, and every pair of bits encodes a turn ranging from 0
        to 4:

        * The first 4 correspond to the first 2 turns in the sequence. These 2 turns can arbitrarily
          be set to any value due to rotation symmetry. Therefore the first 4 bits will be unused.

        * If there is no secondary chain going out from the 2nd bead in the main chain, another
          symmetry argument makes it such that the 3rd turn has effectively only 2 options. Therefore
          the 5th qubit can sometimes be unused as well.

        * The remaining pairs of qubits will encode the remaining turns of the main bead and then the
          turns of the secondary chains in that order.

        Example: In the context of a protein of length 5 with secondary chains in the 2nd and 4th
        position ``10110110`` encodes the most efficient configuration. We start by flipping the
        string and pairing up the bits ``01-10-11-01``. Note that in this case we have an even number
        of bits. This is only due to the fact that we have a secondary chain in the second position.
        Since the first 2 turns on the main chain were arbitrarily set (In qiskit we chose to set
        them to ``[1,0]`` respectively) the sequence of turns in the main chain is ``[0,1,1,2]``.
        The remaining pairs of bits indicate that the turns from the secondary chains in the 2nd
        and 4th position are ``3`` and ``1`` respectively.
        For more information see: `<https://doi.org/10.1038/s41534-021-00368-4>`__.

        """
        unused_qubits = self._unused_qubits
        result = []
        offset = 0
        size = len(self._turn_sequence)
        for i in range(size):
            index = size - 1 - i
            while i + offset in unused_qubits:
                result.append("_")
                offset += 1
            result.append(self._turn_sequence[index])

        return "".join(result[::-1])

    def save_xyz_file(
        self, name: Optional[str] = None, path: str = "", comment: str = "", replace: bool = False
    ) -> None:

        """
        Generates a .xyz file.

        Args:
            name: Name of the file to be generated. If the name is ``None`` the
                name of the file will be the letters of the aminoacids on the main_chain.
                If a file of the same name already exists then the action taken is dependent
                on the `replace` arg.
            path: Path where the file will be generated. If left empty the file will
                be saved in the working directory.
            comment: Comment to be added to the second line of the file. By default, the line will
                be left blank.
            replace: If ``True``, the file will be overwritten if it already exists.
        Raises:
            FileExistsError: If the file already exists and replace is ``False``.
        """
        if name is None:
            name = str(self._peptide.get_main_chain.main_chain_residue_sequence)
        self.protein_shape_file_gen.save_xyz_file(
            name=name, path=path, comment=comment, replace=replace
        )

    @_optionals.HAS_MATPLOTLIB.require_in_call
    def get_figure(
        self, title: str = "Protein Structure", ticks: bool = False, grid: bool = False
    ) -> "figure":
        """
        Generates a figure of the molecule in 3D.

        Args:
            title: The title of the plot.
            ticks: Boolean for showing ticks in the graphic.
            grid: Boolean for showing the grid in the graphic.
        Returns:
            A figure with the folded protein.
        """
        return ProteinPlotter(self.protein_shape_file_gen).get_figure(
            title=title, ticks=ticks, grid=grid
        )
