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

"""Qiskit Nature Settings."""


class QiskitNatureSettings:
    """Global settings for Qiskit Nature."""

    def __init__(self):
        self._dict_aux_operators: bool = False

    @property
    def dict_aux_operators(self) -> bool:
        """Return whether `aux_operators` are dictionary- or list-based."""
        return self._dict_aux_operators

    @dict_aux_operators.setter
    def dict_aux_operators(self, _dict_aux_operators: bool) -> None:
        """Set whether `aux_operators` are dictionary- or list-based."""
        self._dict_aux_operators = _dict_aux_operators


settings = QiskitNatureSettings()
