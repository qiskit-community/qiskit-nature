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
class Penalties:
    def __init__(self, lambda_chiral, lambda_back, lambda_1, lambda_contacts):
        self._lambda_chiral = lambda_chiral
        self._lambda_back = lambda_back
        self._lambda_1 = lambda_1
        self._lambda_contacts = lambda_contacts

    @property
    def lambda_chiral(self):
        return self._lambda_chiral

    @property
    def lambda_back(self):
        return self._lambda_back

    @property
    def lambda_1(self):
        return self._lambda_1

    @property
    def lambda_contacts(self):
        return self._lambda_contacts
