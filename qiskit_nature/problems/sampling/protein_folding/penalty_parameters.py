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
"""Gathers parameters for penalty terms in a protein folding problem Hamiltonian."""


class PenaltyParameters:
    """Gathers parameters for penalty terms in a protein folding problem Hamiltonian."""

    def __init__(
        self,
        penalty_chiral: float = 10.0,
        penalty_back: float = 10.0,
        penalty_1: float = 10.0,
        penalty_contacts: float = 10.0,
    ):
        """
        Args:
            penalty_chiral: A penalty parameter used to impose the right chirality.
            penalty_back: A penalty parameter used to penalize turns along the same axis.
            penalty_1: A penalty parameter used to penalize local overlap between beads within a
                nearest neighbor contact.
            penalty_contacts: A penalty parameter used to penalize local overlap between beads
                within a nearest neighbor contact.
        """

        self._penalty_chiral = penalty_chiral
        self._penalty_back = penalty_back
        self._penalty_1 = penalty_1
        self._penalty_contacts = penalty_contacts

    @property
    def penalty_chiral(self) -> float:
        """Returns a penalty parameter used to impose the right chirality."""
        return self._penalty_chiral

    @property
    def penalty_back(self) -> float:
        """Returns a penalty parameter used to penalize turns along the same axis."""
        return self._penalty_back

    @property
    def penalty_1(self) -> float:
        """Returns a penalty parameter used to penalize local overlap between beads within a
        nearest neighbor contact."""
        return self._penalty_1

    @property
    def penalty_contacts(self) -> float:
        """Returns a penalty parameter used to penalize local overlap between beads within a
        nearest neighbor contact."""
        return self._penalty_contacts
