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

"""The HFInitialPoint class to compute an initial point for the VQE Ansatz parameters."""

from __future__ import annotations

import warnings

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.properties import ElectronicEnergy, GroupedSecondQuantizedProperty

from .initial_point import InitialPoint


class HFInitialPoint(InitialPoint):
    r"""Compute the Hartree-Fock (HF) initial point.

    A class that provides an all-zero initial point for the ``VQE`` parameter values.

    If used in concert with the
    :class:`~qiskit.circuit.library.initial_states.hartree_fock.HartreeFock` initial state (which
    will be prepended to the :class:`~qiskit.circuit.library.ansatzes.ucc.UCC` circuit) the all-zero
    initial point will correspond to the HF initial point.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ansatz: UCC | None = None
        self._reference_energy: float = 0.0
        self._parameters: np.ndarray | None = None

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz.

        The ``excitation_list`` and ``reps`` used by the
        :class:`~qiskit.circuit.library.ansatzes.ucc.UCC` ansatz is obtained to ensure that the
        shape of the initial point is appropriate.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        self._invalidate()
        self._ansatz = ansatz

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty | None:
        """The grouped property.

        The grouped property is not required to compute the HF initial point. If it is provided we
        will attempt to obtain the HF ``reference_energy``.
        """
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        electronic_energy: ElectronicEnergy | None = grouped_property.get_property(ElectronicEnergy)
        if electronic_energy is None:
            warnings.warn(
                "The ElectronicEnergy was not obtained from the grouped_property. "
                "The grouped_property and reference_energy will not be set."
            )
            return

        self._reference_energy = electronic_energy.reference_energy if not None else 0.0
        self._grouped_property = grouped_property

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as an array."""
        if self._parameters is None:
            self.compute()
        return self._parameters

    def compute(
        self,
        ansatz: UCC | None = None,
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
        if grouped_property is not None:
            self.grouped_property = grouped_property

        if ansatz is not None:
            self.ansatz = ansatz

        if self._ansatz is None:
            raise QiskitNatureError(
                "The ansatz property has not been set. "
                "Not enough information has been provided to compute the initial point. "
                "Set the ansatz or call compute with it as an argument. "
            )

        self._compute()

    def _compute(self) -> None:
        """Computes the HF initial point array for a given excitation list.

        In the Hartree-Fock case this is simply an all-zero array.

        Returns:
            An all-zero array with the same length as the excitation list.
        """
        # Ansatz operators must be built to compute the excitation list.
        _ = self._ansatz.operators
        self._parameters = np.zeros(
            self._ansatz.reps * len(self._ansatz.excitation_list), dtype=float
        )

    @property
    def total_energy(self) -> float:
        """The Hartree-Fock reference energy.

        If the reference energy was not obtained from
        :class:`~qiskit_nature.second_q.properties.ElectronicEnergy`
        this will be equal to zero.
        """
        return self._reference_energy

    def _invalidate(self):
        """Invalidate any previous computation."""
        self._parameters = None
