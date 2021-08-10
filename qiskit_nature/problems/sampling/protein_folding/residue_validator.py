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
"""Validates protein residues provided."""
from typing import List, Union

from .exceptions.invalid_residue_exception import (
    InvalidResidueException,
)


def _validate_residue_sequence(residue_sequence: Union[List[str], str]):
    """
    Checks if the provided residue sequence contains allowed characters.

    Args:
        residue_sequence: A list that contains characters defining residues for a chain of proteins.

    Throws:
        InvalidResidueException: If an illegal residue character is discovered.
    """
    for residue_symbol in residue_sequence:
        _validate_residue_symbol(residue_symbol)


def _validate_residue_symbol(residue_symbol: str):
    """
    Checks if the provided residue character is legal. If not, an InvalidResidueException is thrown.

    Args:
        residue_symbol: symbol of a residue.

    Raises:
        InvalidResidueException: if a symbol provided is not legal.
    """
    valid_residues = [
        "A",  # Alanine
        "C",  # Cysteine
        "D",  # Aspartic acid
        "E",  # Glutamic acid
        "F",  # Phenylalanine
        "G",  # Glycine
        "H",  # Histidine
        "I",  # Isoleucine
        "K",  # Lysine
        "L",  # Leucine
        "M",  # Methionine
        "N",  # Asparagine
        "P",  # Proline
        "Q",  # Glutamine
        "R",  # Arginine
        "S",  # Serine
        "T",  # Threonine
        "V",  # Valine
        "W",  # Tryptophan
        "Y",  # Tyrosine
    ]
    if residue_symbol != "" and residue_symbol not in valid_residues:
        raise InvalidResidueException(
            f"Provided residue type {residue_symbol} is not valid. Valid residue types are "
            f"{valid_residues}"
        )
