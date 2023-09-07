# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Nature Settings."""

from __future__ import annotations


class QiskitNatureSettings:
    """Global settings for Qiskit Nature."""

    def __init__(self) -> None:
        self._optimize_einsum = True

        # The set below can be used to handle deprecation warnings for various settings.
        # It exists to keep track of which deprecation warnings were already shown in order to avoid
        # spamming the user with the same warning over and over.
        # To use it, simply add a unique string to this set after having raised some warning and
        # only raise the warning in the first place, if this unique string is not already part of
        # this set.
        self._deprecation_shown: set[str] = set()

    @property
    def optimize_einsum(self) -> bool:
        """Returns the setting used for `numpy.einsum(optimize=...)`.

        This is only used for calls with 3 or more operands. For more details refer to:
        https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        return self._optimize_einsum

    @optimize_einsum.setter
    def optimize_einsum(self, optimize_einsum: bool) -> None:
        """Sets the setting used for `numpy.einsum(optimize=...)`.

        This is only used for calls with 3 or more operands. For more details refer to:
        https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        self._optimize_einsum = optimize_einsum


settings = QiskitNatureSettings()
