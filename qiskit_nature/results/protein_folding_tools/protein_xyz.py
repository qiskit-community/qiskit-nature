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
"""An auxiliary class that gets the coordinates of aminoacids of a molecule
 in ProteinFoldingResult."""
from typing import Union, List, Optional
import numpy as np

from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


class ProteinXYZ:
    """This class handles the creation of cartesian coordinates for
    each aminoacid in a protein and generates a .xyz file.
    It is used by the ProteinFoldingResult class.
    """

    coordinates = (1.0 / np.sqrt(3)) * np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1], [1, -1, 1]])

    def __init__(
        self,
        main_chain_turns: List[int],
        side_chain_turns: List[Union[None, int]],
        peptide: Peptide,
    ) -> None:
        """
        Args:
            main_chain_turns : a list of integers encoding the turns of the main chain.
            side_chain_turns : a list of integers and None encoding the turns of the main chain
                               or None.
            peptide: the peptide we are getting the positions for.

        """
        self._main_chain_turns = main_chain_turns
        self._side_chain_turns = side_chain_turns

        self._main_chain_aminoacid_list = np.array(
            list(peptide.get_main_chain.main_chain_residue_sequence)
        )
        self._side_chain_aminoacid_list = np.array(
            [
                aminoacid.residue_sequence[0] if (aminoacid is not None) else None
                for aminoacid in peptide.get_side_chains()
            ]
        )
        self._main_positions = self.generate_main_positions()
        self._side_positions = self.generate_side_positions()

    def generate_side_positions(self) -> List[Optional[np.ndarray]]:
        """Generates the positions of the side chain."""
        side_positions: List[Optional[np.ndarray]] = []
        counter = 1
        for mainpos, sideturn in zip(self.main_positions, self._side_chain_turns):
            if sideturn is None:
                side_positions.append(None)
            else:
                side_positions.append(mainpos + (-1) ** counter * self.coordinates[sideturn])

            counter += 1
        return side_positions

    @property
    def side_positions(self) -> List[Optional[np.ndarray]]:
        """
        Returns the xyz position for each side chain element.

        Returns:
            A list with the position of the side chain of each bead in the main chain in order.
            None in the i-th position of the list corresponds to no side chain at that
            position of the main chain.

        """

        return self._side_positions

    def generate_main_positions(self) -> np.ndarray:
        """Generates the positions of the main chain."""
        lenght_turns = len(self._main_chain_turns)
        relative_positions = np.zeros((lenght_turns + 1, 3), dtype=float)

        for i in range(lenght_turns):
            relative_positions[i + 1] = (-1) ** i * self.coordinates[self._main_chain_turns[i]]

        return relative_positions.cumsum(axis=0)

    @property
    def main_positions(self) -> np.ndarray:
        """
        Returns an array with the cartesian coordinates of each aminoacid in the main chain.
        The first time called it generates the coordinates.

        Returns:
            An array with the cartesian coordinates of each aminoacid in the main chain.

        """

        return self._main_positions

    def get_xyz_file(self, name: str, output_data: bool) -> np.ndarray:
        """
        Creates a .xyz file and saves it in the current directory.
        """
        main_data = np.column_stack([self._main_chain_aminoacid_list, self.main_positions])

        # We will discard the None values corresponding to empty side chains.
        side_aminoacid = np.array(self._side_chain_aminoacid_list)
        side_aminoacid = side_aminoacid[side_aminoacid != None]
        side_aminoacid = side_aminoacid.astype("<U32")
        
        side_position = np.array(
            [side_pos for side_pos in self.side_positions if side_pos is not None],
            dtype="<U32",
        )

        
        side_data = np.column_stack([side_aminoacid, side_position])
        if side_data.size != 0:

            data = np.append(main_data, side_data, axis=0)

        else:
            data = main_data

        if output_data:
            np.savetxt(name + ".xyz", data, delimiter=" ", fmt="%s")

        return data
