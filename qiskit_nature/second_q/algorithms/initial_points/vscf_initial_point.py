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

    The excitation list generated by the :class:`~qiskit.circuit.library.ansatzes.uvcc.UVCC` ansatz
    is obtained to ensure that the shape of the initial point is appropriate.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ansatz: UVCC | None = None
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None
        self._parameters: np.ndarray | None = None

    @property
    def ansatz(self) -> UVCC | None:
        """The UVCC ansatz.

        This is used to ensure that the :attr:`excitation_list` matches with the UVCC ansatz that
        will be used with the VQE algorithm.

        Raises:
            QiskitNatureError: If not set using a valid
                :class:`~qiskit_nature.second_q.circuit.library.ansatzes.uvcc.UVCC` instance.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UVCC) -> None:
        # Operators must be built early to compute the excitation list.
        _ = ansatz.operators

        # Invalidate any previous computation.
        self._parameters = None

        self._excitation_list = ansatz.excitation_list
        self._ansatz = ansatz

    @property
    def excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The list of excitations.

        Setting this will overwrite the excitation list from the ansatz.
        """
        return self._excitation_list

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
        """Compute the initial point.

        See further up for more information.

        Args:
            ansatz: The UVCC ansatz. Required to set the :attr:`excitation_list` to ensure that the
                    coefficients are mapped correctly in the initial point array.
            grouped_property: Not required to compute the VSCF initial point.

        Raises:
            QiskitNatureError: If :attr`ansatz` is not set.
        """
        if ansatz is not None:
            # The ansatz setter also sets the private excitation_list.
            self.ansatz = ansatz

        if self._excitation_list is None:
            raise QiskitNatureError(
                "The excitation list has not been set directly or via the ansatz. "
                "Not enough information has been provided to compute the initial point. "
                "Set the ansatz or call compute with it as an argument. "
                "The ansatz is not required if the excitation list has been set directly."
            )

        self._parameters = np.zeros(self.ansatz.reps * len(self._excitation_list), dtype=float)
