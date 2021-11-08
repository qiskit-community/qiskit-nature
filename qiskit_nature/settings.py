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
        if not self._dict_aux_operators:
            raise DeprecationWarning(
                "List-based `aux_operators` are deprecated as of version 0.3.0 and support for them"
                " will be removed no sooner than 3 months after the release. Instead, use "
                "dict-based `aux_operators`. You can switch to the dict-based interface "
                "immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`."
            )

        return self._dict_aux_operators

    @dict_aux_operators.setter
    def dict_aux_operators(self, dict_aux_operators: bool) -> None:
        """Set whether `aux_operators` are dictionary- or list-based."""
        if not dict_aux_operators:
            raise DeprecationWarning(
                "List-based `aux_operators` are deprecated as of version 0.3.0 and support for them"
                " will be removed no sooner than 3 months after the release. Instead, use "
                "dict-based `aux_operators`. You can switch to the dict-based interface "
                "immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`."
            )

        self._dict_aux_operators = dict_aux_operators


settings = QiskitNatureSettings()
