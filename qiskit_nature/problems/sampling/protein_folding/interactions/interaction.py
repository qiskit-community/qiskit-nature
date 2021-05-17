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
from abc import ABC, abstractmethod

from problems.sampling.protein_folding.exceptions.invalid_residue_exception import \
    InvalidResidueException


class Interaction(ABC):

    @abstractmethod
    def calc_energy_matrix(self, num_beads: int, sequence):
        pass

    # TODO duplicated from another class
    @staticmethod
    def _validate_residue(sequence):
        valid_residues = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D',
                          'E', 'H', 'R', 'K', 'P']
        for letter in sequence:
            if letter not in valid_residues:
                raise InvalidResidueException(
                    f"Provided residue type {letter} is not valid. Valid residue types are [C, "
                    f"M, F, I, L, V, W, Y, A, G, T, S, N, Q, D, E, H, R, K, P].")
