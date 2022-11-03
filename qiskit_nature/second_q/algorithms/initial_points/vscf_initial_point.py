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
from qiskit_nature.second_q.problems import BaseProblem

from .initial_point import InitialPoint


class VSCFInitialPoint(InitialPoint):
    r"""Compute the vibrational self-consistent field (VSCF) initial point.

    A class that provides an all-zero initial point for the ``VQE`` parameter values.

    If used in concert with the
    :class:`~qiskit_nature.second_q.circuit.library.initial_states.vscf.VSCF`
    initial state (which will be prepended to the
    :class:`~qiskit_nature.second_q.circuit.library.ansatzes.uvcc.UVCC` circuit) the all-zero
    initial point will correspond to the VSCF initial point.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ansatz: UVCC | None = None
        self._parameters: np.ndarray | None = None

    @property
    def ansatz(self) -> UVCC | None:
        """The UVCC ansatz.

        The ``excitation_list`` and ``reps`` used by the
        :class:`~qiskit_nature.second_q.circuit.library.ansatzes.uvcc.UVCC` ansatz is obtained to
        ensure that the shape of the initial point is appropriate.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UVCC) -> None:
        self._invalidate()
        self._ansatz = ansatz

    @property
    def problem(self) -> BaseProblem | None:
        """The problem.

        The problem is not required to compute the VSCF initial point.
        """
        return self._problem

    @problem.setter
    def problem(self, problem: BaseProblem) -> None:
        self._problem = problem

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as an array."""
        if self._parameters is None:
            self.compute()
        return self._parameters

    def compute(
        self,
        ansatz: UVCC | None = None,
        problem: BaseProblem | None = None,
    ) -> None:
        """Compute the initial point parameter for each excitation.

        See class documentation for more information.

        Args:
            problem: The :attr:`problem`.
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

        self._compute()

    def _compute(self) -> None:
        """Computes the VSCF initial point array for a given excitation list.

        In the VSCF case this is simply an all-zero array.

        Returns:
            An all-zero array with the same length as the excitation list.
        """
        # Ansatz operators must be built to compute the excitation list.
        _ = self._ansatz.operators
        self._parameters = np.zeros(
            self._ansatz.reps * len(self._ansatz.excitation_list), dtype=float
        )

    def _invalidate(self):
        """Invalidate any previous computation."""
        self._parameters = None
