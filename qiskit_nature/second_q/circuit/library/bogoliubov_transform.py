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

"""Bogoliubov transform."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import RZGate, XXPlusYYGate
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.utils import apply_matrix_to_slices, givens_matrix
from qiskit_nature.utils.linalg import fermionic_gaussian_decomposition_jw


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
    n, p = mat.shape  # pylint: disable=invalid-name
    if p == n:
        if not _rows_are_orthonormal(mat, rtol=rtol, atol=atol):
            raise ValueError("transformation_matrix must have orthonormal rows.")
    elif p == n * 2:
        left = mat[:, :n]
        right = mat[:, n:]
        comm1 = left @ left.T.conj() + right @ right.T.conj()
        comm2 = left @ right.T + right @ left.T
        if not np.allclose(comm1, np.eye(n), rtol=rtol, atol=atol) or not np.allclose(
            comm2, 0.0, atol=atol
        ):
            raise ValueError(
                "transformation_matrix does not describe a valid transformation "
                "of fermionic ladder operators. A valid matrix should have the block form "
                "[W1 W2] where W1 @ W1.T.conj() + W2 @ W2.T.conj() = I and "
                "W1 @ W2.T + W2 @ W1.T = 0."
            )
    else:
        raise ValueError(
            f"transformation_matrix must be N x N or N x 2N. Instead, got shape {mat.shape}."
        )


class BogoliubovTransform(QuantumCircuit):
    r"""A circuit that performs a Bogoliubov transform.

    A Bogoliubov transform effects a unitary basis change that maps the fermionic ladder
    operators to a new set of ladder operators that also satisfy the fermionic
    anticommutation relations. That is, it effects a unitary :math:`U` such that

    .. math::
        U a^\dagger_j U^\dagger = b^\dagger_j, \quad j = 1, \ldots, N

    where the :math:`\{a_j\}` are the original fermionic creation operators
    and the :math:`\{b_j\}` are the new fermionic creation operators.
    The new creation operators are linear combinations of the original ladder operators,
    and the coefficients of the linear combinations are specified by a matrix :math:`W`
    which determines the unitary :math:`U`. The matrix :math:`W` is either
    :math:`N \times N` or :math:`N \times 2N`.

    If :math:`W` is :math:`N \times N`, then the linear combinations involve only the
    original creation operators:

    .. math::
        \begin{pmatrix}
            b^\dagger_1 \\
            \vdots \\
            b^\dagger_N \\
        \end{pmatrix}
        = W
        \begin{pmatrix}
            a^\dagger_1 \\
            \vdots \\
            a^\dagger_N \\
        \end{pmatrix}.

    If :math:`W` is :math:`N \times 2N`, then the linear combinations involve both the
    original creation and annihilation operators:

    .. math::
        \begin{pmatrix}
            b^\dagger_1 \\
            \vdots \\
            b^\dagger_N \\
        \end{pmatrix}
        = W
        \begin{pmatrix}
            a^\dagger_1 \\
            \vdots \\
            a^\dagger_N \\
            a_1 \\
            \vdots \\
            a_N
        \end{pmatrix}.

    The matrix :math:`W` is commonly obtained by calling the
    :meth:`~.QuadraticHamiltonian.diagonalizing_bogoliubov_transform`
    method of the :class:`~.QuadraticHamiltonian` class.

    Currently, only the Jordan-Wigner Transformation is supported.

    References:
        - `arXiv:1711.05395`_
        - `arXiv:1603.08788`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395
    .. _arXiv:1603.08788: https://arxiv.org/abs/1603.08788
    """

    def __init__(
        self,
        transformation_matrix: np.ndarray,
        qubit_converter: Optional[QubitConverter] = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **circuit_kwargs,
    ) -> None:
        r"""
        Args:
            transformation_matrix: The matrix :math:`W` that specifies the coefficients of the
                new creation operators in terms of the original creation operators.
                Should be either :math:`N \times N` or :math:`N \times 2N`.
            qubit_converter: The qubit converter. The default behavior is to create
                one using the call `QubitConverter(JordanWignerMapper())`.
            validate: Whether to validate the inputs.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.
            circuit_kwargs: Keyword arguments to pass to the QuantumCircuit initializer.

        Raises:
            ValueError: transformation_matrix must be a 2-dimensional array.
            ValueError: transformation_matrix must have orthonormal rows.
            ValueError: transformation_matrix does not describe a valid transformation
                of fermionic ladder operators. If the transformation matrix is
                :math:`N \times N`, then it should be unitary.
                If the transformation matrix is :math:`N \times 2N`, then it should have the block form
                :math:`(W_1 \quad W_2)` where :math:`W_1 W_1^\dagger + W_2 W_2^\dagger = I` and
                :math:`W_1 W_2^T + W_2 W_1^T = 0`.
            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
                Please use
                :class:`qiskit_nature.second_q.mappers.JordanWignerMapper`
                to construct the qubit mapper.
        """
        if validate:
            _validate_transformation_matrix(transformation_matrix, rtol=rtol, atol=atol)

        if qubit_converter is None:
            qubit_converter = QubitConverter(JordanWignerMapper())

        n, _ = transformation_matrix.shape
        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            operations = _bogoliubov_transform_jw(register, transformation_matrix)
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use "
                "qiskit_nature.second_q.mappers.JordanWignerMapper "
                "to construct the qubit mapper."
            )


def _bogoliubov_transform_jw(
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
    n, p = transformation_matrix.shape  # pylint: disable=invalid-name
    if p == n:
        yield from _bogoliubov_transform_num_conserving_jw(register, transformation_matrix)
    else:
        yield from _bogoliubov_transform_general_jw(register, transformation_matrix)


def _bogoliubov_transform_num_conserving_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
    n, _ = transformation_matrix.shape
    current_matrix = transformation_matrix
    left_rotations = []
    right_rotations = []

    # compute left and right Givens rotations
    for i in range(n - 1):
        if i % 2 == 0:
            # rotate columns by right multiplication
            for j in range(i + 1):
                target_index = i - j
                row = n - j - 1
                if not np.isclose(current_matrix[row, target_index], 0.0):
                    # zero out element at target index in given row
                    givens_mat = givens_matrix(
                        current_matrix[row, target_index + 1],
                        current_matrix[row, target_index],
                    )
                    right_rotations.append((givens_mat, (target_index + 1, target_index)))
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat,
                        [(Ellipsis, target_index + 1), (Ellipsis, target_index)],
                    )
        else:
            # rotate rows by left multiplication
            for j in range(i + 1):
                target_index = n - i + j - 1
                col = j
                if not np.isclose(current_matrix[target_index, col], 0.0):
                    # zero out element at target index in given column
                    givens_mat = givens_matrix(
                        current_matrix[target_index - 1, col],
                        current_matrix[target_index, col],
                    )
                    left_rotations.append((givens_mat, (target_index - 1, target_index)))
                    current_matrix = apply_matrix_to_slices(
                        current_matrix, givens_mat, [target_index - 1, target_index]
                    )

    # convert left rotations to right rotations
    for givens_mat, (i, j) in reversed(left_rotations):
        givens_mat = givens_mat.T.conj()
        givens_mat[:, 0] *= current_matrix[i, i]
        givens_mat[:, 1] *= current_matrix[j, j]
        new_givens_mat = givens_matrix(givens_mat[1, 1], givens_mat[1, 0])
        right_rotations.append((new_givens_mat.T, (i, j)))
        phase_matrix = givens_mat @ new_givens_mat
        current_matrix[i, i] = phase_matrix[0, 0]
        current_matrix[j, j] = phase_matrix[1, 1]

    # yield operations
    for i in range(n):
        phi = np.angle(current_matrix[i, i])
        yield RZGate(phi), (register[i],)
    for givens_mat, (i, j) in reversed(right_rotations):
        theta = np.arccos(np.real(givens_mat[0, 0]))
        phi = np.angle(givens_mat[0, 1])
        yield XXPlusYYGate(2 * theta, phi - np.pi / 2), (register[j], register[i])


def _bogoliubov_transform_general_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
    decomposition, left_unitary = fermionic_gaussian_decomposition_jw(
        register, transformation_matrix
    )
    yield from _bogoliubov_transform_num_conserving_jw(register, left_unitary.T)
    yield from reversed(decomposition)
