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
class PenaltyParameters:

    def __init__(self, lambda_chiral: float = 10, lambda_back: float = 10, lambda_1: float = 10,
                 lambda_contacts: float = 10):
        self._lambda_chiral = lambda_chiral
        self._lambda_back = lambda_back
        self._lambda_1 = lambda_1
        self._lambda_contacts = lambda_contacts
    @property
    def lambda_chiral(self):
        """Returns a penalty parameter used to impose the right chirality."""
        return self._lambda_chiral

    @property
    def lambda_back(self):
        """Returns a penalty parameter used to penalize turns along the same axis."""
        return self._lambda_back

    @property
    def lambda_1(self):
        """Returns a penalty parameter used to penalize local overlap between beads within a nearest neighbor contact."""
        return self._lambda_1

    @property
    def lambda_contacts(self):
        """Returns a penalty parameter used to penalize local overlap between beads within a nearest neighbor contact."""
        return self._lambda_contacts
