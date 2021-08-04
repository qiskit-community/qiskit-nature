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
"""A class that stores contacts between beads of a peptide as qubit operators."""
from typing import Dict

from .contact_map_builder import (
    _create_contact_qubits,
)
from ..peptide.peptide import Peptide


class ContactMap:
    """A class that stores contacts between beads of a peptide as qubit operators. For technical
    details regarding the meaning of these operators as well as a convention for their indexing,
    please see the documentation in the ContactMapBuilder class."""

    def __init__(self, peptide: Peptide):
        """
        Args:
            peptide: A Peptide object that includes all information about a protein.
        """
        self._peptide = peptide
        (
            self._lower_main_upper_main,
            self._lower_side_upper_main,
            self._lower_main_upper_side,
            self._lower_side_upper_side,
            self.num_contacts,
        ) = _create_contact_qubits(peptide)

    @property
    def peptide(self) -> Peptide:
        """Returns a peptide."""
        return self._peptide

    @property
    def lower_main_upper_main(self) -> Dict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a bead on a main chain (first index in a dictionary) and a bead in a main chain (
        second index in a dictionary)."""
        return self._lower_main_upper_main

    @property
    def lower_side_upper_main(self) -> Dict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a first bead in a side chain (first index in a dictionary) and a bead in a main
        chain (second index in a dictionary)."""
        return self._lower_side_upper_main

    @property
    def lower_main_upper_side(self) -> Dict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a bead in a main chain (first index in a dictionary) and a first bead in a side
        chain (second index in a dictionary)."""
        return self._lower_main_upper_side

    @property
    def lower_side_upper_side(self) -> Dict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a first bead in a side chain (first index in a dictionary) and a first bead in a
        side chain (second index in a dictionary)."""
        return self._lower_side_upper_side
