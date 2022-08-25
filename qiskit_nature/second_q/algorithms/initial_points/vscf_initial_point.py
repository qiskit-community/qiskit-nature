# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The VSCFInitialPoint class to compute an initial point for the VQE Ansatz parameters."""

from __future__ import annotations

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.circuit.library import UVCC
from qiskit_nature.second_q.properties.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)

from .initial_point import InitialPoint


class VSCFInitialPoint(InitialPoint):
    r"""Compute the vibrational self-consistent field (VSCF) initial point.

    A class that provides an all-zero initial point for the ``VQE`` parameter values.

    If used in concert with the :class:`~qiskit.circuit.library.initial_states.vscf.VSCF` initial
    state (which will be prepended to the :class:`~qiskit.circuit.library.ansatzes.uvcc.UVCC`
    circuit) the all-zero initial point will correspond to the VSCF initial point.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ansatz: UVCC | None = None
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None
        self._reps: int = 1
        self._parameters: np.ndarray | None = None

    @property
    def ansatz(self) -> UVCC | None:
        """The UVCC ansatz.

        The ``excitation_list`` and ``reps`` used by the
        :class:`~qiskit.circuit.library.ansatzes.uvcc.UVCC` ansatz is obtained to ensure that the
        shape of the initial point is appropriate.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UVCC) -> None:
        # Operators must be built early to compute the excitation list.
        _ = ansatz.operators

        self._invalidate()

        self._excitation_list = ansatz.excitation_list
        self._ansatz = ansatz

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty | None:
        """The grouped property.

        The grouped property is not required to compute the VSCF initial point.
        """
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        self._grouped_property = grouped_property

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as an array."""
        if self._parameters is None:
            self.compute()
        return self._parameters

    def compute(
        self,
        ansatz: UVCC | None = None,
        grouped_property: GroupedSecondQuantizedProperty | None = None,
    ) -> None:
        """Compute the initial point parameter for each excitation.

        See class documentation for more information.

        Args:
            grouped_property: The :attr:`grouped_property`.
            ansatz: The :attr:`ansatz`.

        Raises:
            QiskitNatureError: If :attr:`ansatz` is not set.
        """
        if ansatz is not None:
            # The ansatz setter also sets the private excitation_list.
            self.ansatz = ansatz

        if self._ansatz is None:
            raise QiskitNatureError(
                "The ansatz property has not been set. "
                "Not enough information has been provided to compute the initial point. "
                "Set the ansatz or call compute with it as an argument. "
            )

        self._parameters = np.zeros(self.ansatz.reps * len(self._excitation_list), dtype=float)

    def _invalidate(self):
        """Invalidate any previous computation."""
        self._parameters = None
