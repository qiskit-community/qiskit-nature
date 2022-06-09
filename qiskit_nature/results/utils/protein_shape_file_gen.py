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
"""An auxiliary class that gets the coordinates of aminoacids of a molecule
 in ProteinFoldingResult."""
import os
from typing import Union, List, Optional
import numpy as np
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


class ProteinShapeFileGen:
    """This class handles the creation of cartesian coordinates for
    each aminoacid in a protein and generates a .xyz file.
    It is used by :class:`~qiskit_nature.results.ProteinFoldingResult`.
    """

    # Coordinates of the 4 edges of a tetrahedron centered at 0. The vectors are normalized.
    COORDINATES = (1.0 / np.sqrt(3)) * np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1], [1, -1, 1]])

    def __init__(
        self,
        main_chain_turns: List[int],
        side_chain_turns: List[Union[None, int]],
        peptide: Peptide,
    ) -> None:
        """
        Args:
            main_chain_turns: A list of integers encoding the turns of the main chain.
            side_chain_turns: A list of integers and None encoding the turns of the main chain
                or None.
            peptide: The peptide we are getting the positions for.

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
        """
        Generates the positions of the side chain.
        Returns:
            A list of arrays with the cartesian coordinates of the side chain.
        """
        side_positions: List[Optional[np.ndarray]] = []
        counter = 1
        for mainpos, sideturn in zip(self.main_positions, self._side_chain_turns):
            if sideturn is None:
                side_positions.append(None)
            else:
                side_positions.append(mainpos + (-1) ** counter * self.COORDINATES[sideturn])

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
        """
        Generates the positions of the main chain.

        Returns:
            An array with the cartesian coordinates of the main chain.
        """
        length_turns = len(self._main_chain_turns)
        relative_positions = np.zeros((length_turns + 1, 3), dtype=float)

        for i in range(length_turns):
            relative_positions[i + 1] = (-1) ** i * self.COORDINATES[self._main_chain_turns[i]]

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

    def save_xyz_file(
        self, name: str, path: str = "", comment: str = "", overwrite: bool = False
    ) -> None:
        """
        Saves the data as an .xyz file.
        For more information about .xyz files see:
        https://en.wikipedia.org/wiki/XYZ_file_format.

        Args:
            name: The file will be called `"name".xyz`. Can overwrite files.
            path: Path under which the file will be saved. If no path is specified the file will
                be saved in the current working directory.
            comment: Comment to be added to the second line of the file. By default the line will
                be left blank.
        """
        file_path = os.path.join(path, name + ".xyz")
        if os.path.exists(file_path):
            raise FileExistsError(f"File {file_path} already exists.")
        data = self.get_xyz_file()
        number_of_particles = data.shape[0]
        header = f"{number_of_particles}\n{comment}"
        np.savetxt(
            fname=file_path,
            header=header,
            X=data,
            delimiter=" ",
            fmt="%s",
            comments="",
        )

    def get_xyz_file(self) -> np.ndarray:
        """
        Returns an array with the symbols of the atoms and their cartesian coordinates.
        Returns:
            An array with the symbols of the atoms and their cartesian coordinates.
        """
        main_data = np.column_stack([self._main_chain_aminoacid_list, self.main_positions])

        # We will discard the None values corresponding to empty side chains.
        side_aminoacid = np.array(self._side_chain_aminoacid_list)
        side_aminoacid = side_aminoacid[side_aminoacid != np.array(None)]
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

        return data
