# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Eigenstate results module."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from qiskit_algorithms import AlgorithmResult, EigensolverResult, MinimumEigensolverResult
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _statevector_to_circuit(state: Statevector) -> QuantumCircuit:
    circ = QuantumCircuit(state.num_qubits)
    circ.initialize(state, circ.qubits)
    return circ


class EigenstateResult(AlgorithmResult):
    """The eigenstate result interface.

    The following attributes can be read and updated once the ``EigenstateResult`` object has
    been constructed.

    Attributes:
        eigenvalues (np.ndarray | None): the computed eigenvalues.
        eigenstates (list[tuple[QuantumCircuit, Sequence[float] | None]] | None): the eigenstates
            belonging to each of the computed eigenvalues.
        aux_operators_evaluated (list[ListOrDict[complex]] | None): the evaluated aux operators.
        raw_result (AlgorithmResult | None): the raw result, wrapped by this ``EigenstateResult``.
        formatting_precision (int): the number of decimal places to use when formatting the result
            for printing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.eigenvalues: np.ndarray | None = None
        self.eigenstates: list[tuple[QuantumCircuit, Sequence[float] | None]] | None = None
        self.aux_operators_evaluated: list[ListOrDict[complex]] | None = None
        self.raw_result: AlgorithmResult | None = None
        self.formatting_precision: int = 12

    @property
    def groundenergy(self) -> float | None:
        """Returns the lowest eigenvalue."""
        energies = self.eigenvalues
        if isinstance(energies, np.ndarray) and energies.size:
            return energies[0].real
        return None

    @property
    def groundstate(self) -> tuple[QuantumCircuit, Sequence[float] | None] | None:
        """Returns the lowest eigenstate."""
        states = self.eigenstates
        if states:
            return states[0]
        return None

    @classmethod
    def from_result(
        cls, raw_result: EigenstateResult | EigensolverResult | MinimumEigensolverResult
    ) -> EigenstateResult:
        """Constructs an `EigenstateResult` from another result type.

        Args:
            raw_result: the raw result from which to build the new one.

        Raises:
            TypeError: when an unsupported result type is provided as input.

        Returns:
            The constructed `EigenstateResult`.
        """
        # NOTE: the following inspection of class names is a work-around to handle backwards
        # compatibility for the deprecated qiskit.algorithms module without delving into complex
        # type unions. This logic can be removed once the qiskit.algorithms module no longer wants
        # to be supported.
        cls_names = {cls.__name__ for cls in raw_result.__class__.mro()}

        if isinstance(raw_result, EigenstateResult) or "EigenstateResult" in cls_names:
            return raw_result
        if isinstance(raw_result, EigensolverResult) or "EigensolverResult" in cls_names:
            return EigenstateResult.from_eigensolver_result(raw_result)
        if (
            isinstance(raw_result, MinimumEigensolverResult)
            or "MinimumEigensolverResult" in cls_names
        ):
            return EigenstateResult.from_minimum_eigensolver_result(raw_result)
        raise TypeError(
            f"Cannot construct an EigenstateResult from a result of type, {type(raw_result)}."
        )

    @classmethod
    def from_eigensolver_result(cls, raw_result: EigensolverResult) -> EigenstateResult:
        """Constructs an `EigenstateResult` from an
        :class:`~qiskit_algorithms.EigensolverResult`.

        Args:
            raw_result: the raw result from which to build the `EigenstateResult`.

        Returns:
            The constructed `EigenstateResult`.
        """
        result = EigenstateResult()
        result.raw_result = raw_result
        result.eigenvalues = np.asarray(raw_result.eigenvalues)

        if hasattr(raw_result, "eigenstates"):
            result.eigenstates = [
                (_statevector_to_circuit(Statevector(state)), None)
                for state in raw_result.eigenstates
            ]
        elif hasattr(raw_result, "optimal_circuits") and hasattr(raw_result, "optimal_points"):
            result.eigenstates = list(zip(raw_result.optimal_circuits, raw_result.optimal_points))

        if raw_result.aux_operators_evaluated is not None:
            result.aux_operators_evaluated = [
                cls._unwrap_aux_op_values(aux_op_eval)
                for aux_op_eval in raw_result.aux_operators_evaluated
            ]

        return result

    @classmethod
    def from_minimum_eigensolver_result(
        cls, raw_result: MinimumEigensolverResult
    ) -> EigenstateResult:
        """Constructs an `EigenstateResult` from an
        :class:`~qiskit_algorithms.MinimumEigensolverResult`.

        Args:
            raw_result: the raw result from which to build the `EigenstateResult`.

        Returns:
            The constructed `EigenstateResult`.
        """
        result = EigenstateResult()
        result.raw_result = raw_result
        result.eigenvalues = np.asarray([raw_result.eigenvalue])

        if hasattr(raw_result, "eigenstate"):
            result.eigenstates = [
                (_statevector_to_circuit(Statevector(raw_result.eigenstate)), None)
            ]
        elif hasattr(raw_result, "optimal_circuit") and hasattr(raw_result, "optimal_point"):
            result.eigenstates = [(raw_result.optimal_circuit, raw_result.optimal_point)]

        if raw_result.aux_operators_evaluated is not None:
            result.aux_operators_evaluated = [
                cls._unwrap_aux_op_values(raw_result.aux_operators_evaluated)
            ]

        return result

    @staticmethod
    def _unwrap_aux_op_values(
        aux_operators_evaluated: ListOrDict[tuple[complex, dict[str, Any]]]
    ) -> ListOrDict[complex]:
        aux_op_values: ListOrDict[complex]
        if isinstance(aux_operators_evaluated, list):
            aux_op_values = [val[0] for val in aux_operators_evaluated]
        else:
            aux_op_values = {key: val[0] for key, val in aux_operators_evaluated.items()}
        return aux_op_values
