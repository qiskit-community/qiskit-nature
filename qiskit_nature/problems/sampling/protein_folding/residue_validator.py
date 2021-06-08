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
from problems.sampling.protein_folding.exceptions.invalid_residue_exception import \
    InvalidResidueException


def _validate_residue_sequence(residue_sequence):
    for residue_symbol in residue_sequence:
        _validate_residue_symbol(residue_symbol)


def _validate_residue_symbol(residue_symbol: str):
    valid_residues = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D',
                      'E', 'H', 'R', 'K', 'P']
    if residue_symbol is not None and residue_symbol not in valid_residues:
        raise InvalidResidueException(
            f"Provided residue type {residue_symbol} is not valid. Valid residue types are [C, "
            f"M, F, I, L, V, W, Y, A, G, T, S, N, Q, D, E, H, R, K, P].")
