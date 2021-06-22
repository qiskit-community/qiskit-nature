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
from typing import DefaultDict, List, Union

from qiskit.opflow import PauliSumOp, PauliOp

from .contact_map_builder import (
    _create_contact_qubits,
)
from ..peptide.pauli_ops_builder import _build_full_identity
from ..peptide.peptide import Peptide


class ContactMap:
    """A class that stores contacts between beads of a peptide as qubit operators."""

    def __init__(self, peptide: Peptide):
        self._peptide = peptide
        (
            self._lower_main_upper_main,
            self._lower_side_upper_main,
            self._lower_main_upper_side,
            self._lower_side_upper_side,
            self.num_contacts,
        ) = _create_contact_qubits(peptide)

        """

       Args:
           peptide: A Peptide object that includes all information about a protein.
       """

    @property
    def peptide(self) -> Peptide:
        """Returns a peptide."""
        return self._peptide

    @property
    def lower_main_upper_main(self) -> DefaultDict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a bead on a main chain (first index) and a bead in a main chain (second index)."""
        return self._lower_main_upper_main

    @property
    def lower_side_upper_main(self) -> DefaultDict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a first bead in a side chain (first index) and a bead in a main chain (second
        index)."""
        return self._lower_side_upper_main

    @property
    def lower_main_upper_side(self) -> DefaultDict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a bead in a main chain (first index) and a first bead in a side chain (second
        index)."""
        return self._lower_main_upper_side

    @property
    def lower_side_upper_side(self) -> DefaultDict[int, dict]:
        """Returns a dictionary which is a component of a contact map that stores contact operators
        between a first bead in a side chain (first index) and a a first bead in a side chain (
        second index)."""
        return self._lower_side_upper_side

    def _create_peptide_qubit_list(self):
        """
        Creates new set of contact qubits for second nearest neighbor
        interactions. Note, the need of multiple interaction qubits
        for each i,j pair.

        Returns:
            new_qubits: The list of all qubits.
        """
        main_chain_len = len(self.peptide.get_main_chain)
        side_chain = self.peptide.get_side_chain_hot_vector()
        old_qubits_conf = []
        old_qubits_contact = []
        num_qubits = 2 * (main_chain_len - 1)
        full_id = _build_full_identity(num_qubits)
        for q in range(3, main_chain_len):
            if q != 3:
                old_qubits_conf.append(full_id ^ self.peptide.get_main_chain[q - 1].turn_qubits[0])
                old_qubits_conf.append(full_id ^ self.peptide.get_main_chain[q - 1].turn_qubits[1])
            else:
                old_qubits_conf.append(full_id ^ self.peptide.get_main_chain[q - 1].turn_qubits[0])
            if side_chain[q - 1]:
                old_qubits_conf.append(
                    self.peptide.get_main_chain[q - 1].side_chain[0].turn_qubits[0] ^ full_id
                )
                old_qubits_conf.append(
                    self.peptide.get_main_chain[q - 1].side_chain[0].turn_qubits[1] ^ full_id
                )

        self._add_qubits(main_chain_len, old_qubits_contact, self._lower_main_upper_main)
        self._add_qubits(main_chain_len, old_qubits_contact, self._lower_side_upper_main)
        self._add_qubits(main_chain_len, old_qubits_contact, self._lower_main_upper_side)
        self._add_qubits(main_chain_len, old_qubits_contact, self._lower_side_upper_side)

        new_qubits = [0] + old_qubits_conf + old_qubits_contact
        return new_qubits

    @staticmethod
    def _add_qubits(
        main_chain_len: int,
        contact_qubits: List[Union[PauliSumOp, PauliOp]],
        contact_map_component: DefaultDict[int, dict],
    ):
        for lower_bead_id in range(1, main_chain_len - 3):
            for upper_bead_id in range(lower_bead_id + 4, main_chain_len + 1):
                try:
                    contact_op = contact_map_component[lower_bead_id][upper_bead_id]
                    contact_qubits.append(contact_op)
                except KeyError:
                    pass
