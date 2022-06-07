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

"""Linear algebra utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import RZGate, XGate, XXPlusYYGate

# HACK: Sphinx fails to handle "ellipsis"
# See https://github.com/python/typing/issues/684
if TYPE_CHECKING:
    _SliceAtom = Union[int, slice, "ellipsis"]
else:
    _SliceAtom = Union[int, slice, type(Ellipsis)]

_Slice = Union[_SliceAtom, Tuple[_SliceAtom, ...]]


def apply_matrix_to_slices(
    target: np.ndarray, matrix: np.ndarray, slices: Sequence[_Slice]
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor.

    Args:
        target: The tensor containing the slices on which to apply the matrix.
        matrix: The matrix to apply to slices of the target tensor.
        slices: The slices of the target tensor on which to apply the matrix.

    Returns:
        The tensor resulting from applying the matrix to the slices of the target tensor.
    """
    result = target.copy()
    for i, slice_i in enumerate(slices):
        result[slice_i] *= matrix[i, i]
        for j, slice_j in enumerate(slices):
            if j != i:
                result[slice_i] += target[slice_j] * matrix[i, j]
    return result


def givens_matrix(a: complex, b: complex) -> np.ndarray:
    r"""Compute the Givens rotation to zero out a row entry.

    Returns a :math:`2 \times 2` unitary matrix G that satisfies

    .. math::
        G
        \begin{pmatrix}
            a \\
            b
        \end{pmatrix}
        =
        \begin{pmatrix}
            0 \\
            r
        \end{pmatrix}

    where :math:`r` is a complex number.
    The first column of :math:`G` is guaranteed to contain only real numbers.

    Args:
        a: A complex number representing the first row entry
        b: A complex number representing the second row entry

    Returns:
        The Givens rotation matrix.
    """
    # Handle case that a is zero
    if np.isclose(a, 0.0):
        cosine = 1.0
        sine = 0.0
        phase: complex = 1.0
    # Handle case that b is zero and a is nonzero
    elif np.isclose(b, 0.0):
        cosine = 0.0
        sine = 1.0
        phase = 1.0
    # Handle case that a and b are both nonzero
    else:
        hypotenuse = np.hypot(abs(a), abs(b))
        cosine = abs(b) / hypotenuse
        sine = abs(a) / hypotenuse
        sign_b = b / abs(b)
        sign_a = a / abs(a)
        phase = sign_a * sign_b.conjugate()
        # If phase is a real number, convert it to a float
        if np.isreal(phase):
            phase = np.real(phase)

    return np.array([[cosine, -phase * sine], [sine, phase * cosine]])


def fermionic_gaussian_decomposition_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> tuple[list[tuple[Gate, tuple[Qubit, ...]]], np.ndarray]:
    """Matrix decomposition to prepare a fermionic Gaussian state under the Jordan-Wigner Transform.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395

    Args:
        register: The register containing the qubits to use
        transformation_matrix: The transformation matrix describing
            the fermionic Gaussian state

    Returns:
        - givens rotation decomposition
        - left unitary left over from the decomposition
    """
    n, _ = transformation_matrix.shape

    # transform matrix to align with exposition in arXiv:1711.05395
    left = transformation_matrix[:, :n]
    right = transformation_matrix[:, n:]
    current_matrix = np.block([right.conj(), left.conj()])

    # compute left_unitary
    left_unitary = np.eye(n, dtype=complex)
    for j in range(n - 1):
        # Zero out entries in column k
        for i in range(n - 1 - j):
            # Zero out entry in row l if needed
            if not np.isclose(current_matrix[i, j], 0.0):
                givens_mat = givens_matrix(current_matrix[i, j], current_matrix[i + 1, j])
                current_matrix = apply_matrix_to_slices(current_matrix, givens_mat, [i, i + 1])
                left_unitary = apply_matrix_to_slices(left_unitary, givens_mat, [i, i + 1])

    # decompose matrix into Givens rotations and particle-hole transformations
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    for i in range(n):
        # zero out the columns in row i
        for j in range(n - 1 - i, n):
            if not np.isclose(current_matrix[i, j], 0.0):
                if j == n - 1:
                    # particle-hole transformation
                    decomposition.append((XGate(), (register[-1],)))
                    _swap_columns(current_matrix, n - 1, 2 * n - 1)
                else:
                    # compute Givens rotation
                    givens_mat = givens_matrix(current_matrix[i, j], current_matrix[i, j + 1])
                    theta = np.arcsin(np.real(givens_mat[1, 0]))
                    phi = -np.angle(givens_mat[1, 1])
                    # add operations
                    decomposition.append((RZGate(phi), (register[j + 1],)))
                    decomposition.append(
                        (XXPlusYYGate(2 * theta, -np.pi / 2), (register[j], register[j + 1]))
                    )
                    # update matrix
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat,
                        [(Ellipsis, j), (Ellipsis, j + 1)],
                    )
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat.conj(),
                        [(Ellipsis, n + j), (Ellipsis, n + j + 1)],
                    )

    for i in range(n):
        left_unitary[i] *= current_matrix[i, n + i].conj()

    return decomposition, left_unitary


def _swap_columns(matrix: np.ndarray, i: int, j: int) -> None:
    """Swap columns of a matrix, mutating it."""
    column_i = matrix[:, i].copy()
    column_j = matrix[:, j].copy()
    matrix[:, i], matrix[:, j] = column_j, column_i
