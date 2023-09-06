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

"""Slater determinants."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.mappers import JordanWignerMapper

from .utils.givens_rotations import _prepare_slater_determinant_jw


def _rows_are_orthonormal(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    m, _ = mat.shape
    return np.allclose(mat @ mat.T.conj(), np.eye(m), rtol=rtol, atol=atol)


def _validate_transformation_matrix(
    mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    if not len(mat.shape) == 2:
        raise ValueError(
            "transformation_matrix must be a 2-dimensional array. "
            f"Instead, got shape {mat.shape}."
        )
    if not _rows_are_orthonormal(mat, rtol=rtol, atol=atol):
        raise ValueError("transformation_matrix must have orthonormal rows.")


class SlaterDeterminant(QuantumCircuit):
    r"""A circuit that prepares a Slater determinant.

    A Slater determinant is a state of the form

    .. math::
        b^\dagger_1 \cdots b^\dagger_{N_f} \lvert \text{vac} \rangle,

    where

    .. math::
        b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k.

    - :math:`Q` is an :math:`N_f \times N` matrix with orthonormal rows.
    - :math:`a^\dagger_1, \ldots, a^\dagger_{N}` are the fermionic creation operators.
    - :math:`\lvert \text{vac} \rangle` is the vacuum state.
      (mutual 0-eigenvector of the fermionic number operators :math:`\{a^\dagger_j a_j\}`)

    The matrix :math:`Q` can be obtained by calling the
    :meth:`~.QuadraticHamiltonian.diagonalizing_bogoliubov_transform`
    method of the :class:`~.QuadraticHamiltonian` class when the
    quadratic Hamiltonian conserves particle number.
    This matrix is used to create circuits that prepare eigenstates of the
    quadratic Hamiltonian.

    Currently, only the Jordan-Wigner transformation is supported.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395
    """

    def __init__(
        self,
        transformation_matrix: np.ndarray,
        qubit_mapper: QubitMapper | None = None,
        *,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **circuit_kwargs,
    ) -> None:
        # pylint: disable=unused-argument
        r"""
        Args:
            transformation_matrix: The matrix :math:`Q` that specifies the coefficients of the
                new creation operators in terms of the original creation operators.
                The rows of the matrix must be orthonormal.
            qubit_mapper: The ``QubitMapper``. The default behavior is to create one using the call
                ``JordanWignerMapper()``.
            validate: Whether to validate the inputs.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.
            circuit_kwargs: Keyword arguments to pass to the ``QuantumCircuit`` initializer.

        Raises:
            ValueError: transformation_matrix must be a 2-dimensional array.
            ValueError: transformation_matrix must have orthonormal rows.
            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
                Please use the :class:`qiskit_nature.second_q.mappers.JordanWignerMapper`.
        """
        if validate:
            _validate_transformation_matrix(transformation_matrix, rtol=rtol, atol=atol)

        if qubit_mapper is None:
            qubit_mapper = JordanWignerMapper()

        _, n = transformation_matrix.shape
        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_mapper, JordanWignerMapper):
            operations = _prepare_slater_determinant_jw(register, transformation_matrix)
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use the qiskit_nature.second_q.mappers.JordanWignerMapper."
            )
