# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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
from qiskit.circuit.library import XGate, XXPlusYYGate

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
            r \\
            0
        \end{pmatrix}

    where :math:`r` is a complex number.

    References:
        - `<https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation>`_
        - `<https://www.netlib.org/lapack/lawnspdf/lawn148.pdf>`_

    Args:
        a: A complex number representing the first row entry
        b: A complex number representing the second row entry

    Returns:
        The Givens rotation matrix.
    """
    # Handle case that a is zero
    if np.isclose(a, 0.0):
        cosine = 0.0
        sine = 1.0
    # Handle case that b is zero and a is nonzero
    elif np.isclose(b, 0.0):
        cosine = 1.0
        sine = 0.0
    # Handle case that a and b are both nonzero
    else:
        hypotenuse = np.hypot(abs(a), abs(b))
        cosine = abs(a) / hypotenuse
        sign_a = a / abs(a)
        sine = sign_a * b.conjugate() / hypotenuse

    return np.array([[cosine, sine], [-sine.conjugate(), cosine]])


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
    current_matrix = np.block([right.conj(), left.conj()]).astype(complex, copy=False)

    # compute left_unitary
    left_unitary = np.eye(n, dtype=complex)
    for j in range(n - 1):
        # Zero out entries in column j
        for i in range(n - 1 - j):
            # Zero out entry in row i if needed
            if not np.isclose(current_matrix[i, j], 0.0):
                givens_mat = givens_matrix(current_matrix[i + 1, j], current_matrix[i, j])
                current_matrix = apply_matrix_to_slices(current_matrix, givens_mat, [i + 1, i])
                left_unitary = apply_matrix_to_slices(left_unitary, givens_mat, [i + 1, i])

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
                    givens_mat = givens_matrix(current_matrix[i, j + 1], current_matrix[i, j])
                    theta = np.arccos(np.real(givens_mat[0, 0]))
                    phi = np.angle(givens_mat[0, 1])
                    # add operations
                    decomposition.append(
                        (XXPlusYYGate(2 * theta, phi - np.pi / 2), (register[j], register[j + 1]))
                    )
                    # update matrix
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat,
                        [(Ellipsis, j + 1), (Ellipsis, j)],
                    )
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat.conj(),
                        [(Ellipsis, n + j + 1), (Ellipsis, n + j)],
                    )

    for i in range(n):
        left_unitary[i] *= current_matrix[i, n + i].conj()

    return decomposition, left_unitary


def _swap_columns(matrix: np.ndarray, i: int, j: int) -> None:
    """Swap columns of a matrix, mutating it."""
    column_i = matrix[:, i].copy()
    column_j = matrix[:, j].copy()
    matrix[:, i], matrix[:, j] = column_j, column_i


def modified_cholesky(
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_rank: int | None = None,
    validate: bool = True,
    atol: float = 1e-8,
) -> np.ndarray:
    r"""Modified Cholesky decomposition of a two-body tensor.

    The modified Cholesky decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::
        h_{pqrs} = \sum_{t} L^{(t)}_{pq} L^{(t)}_{rs}

    The number of terms :math:`t` in the decomposition depends on the allowed
    error threshold. A larger error threshold leads to a smaller number of terms.
    Furthermore, the `max_rank` parameter specifies an optional upper bound
    on :math:`t`.

    References:
        - `arXiv:1711.02242`_

    Args:
        two_body_tensor: The two-body tensor to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_rank: The maximum number of terms to include in the decomposition.
        validate: Whether to check that the input tensor has the correct symmetries.
            It should be real and symmetric.
        atol: Absolute numerical tolerance for input validation.

    Returns:
        The Cholesky terms L^{(t)} as a list of matrices assembled into a Numpy array.

    Raises:
        ValueError: The input tensor does not have the correct symmetries.

    .. _arXiv:1711.02242: https://arxiv.org/abs/1711.02242
    """
    n_modes, _, _, _ = two_body_tensor.shape
    reshaped_tensor = np.reshape(two_body_tensor, (n_modes**2, n_modes**2))

    if validate:
        if not np.all(np.isreal(reshaped_tensor)):
            raise ValueError("Two-body tensor must be real.")
        if not np.allclose(reshaped_tensor, reshaped_tensor.T, atol=atol):
            raise ValueError("Two-body tensor must be symmetric.")

    if max_rank is None:
        max_rank = n_modes * (n_modes + 1) // 2

    cholesky_vecs = np.zeros((max_rank + 1, n_modes**2))
    errors = np.diagonal(reshaped_tensor).copy()
    for index in range(max_rank + 1):
        max_error_index = np.argmax(errors)
        max_error = errors[max_error_index]
        if max_error < error_threshold:
            break
        cholesky_vecs[index] = reshaped_tensor[:, max_error_index]
        if index:
            cholesky_vecs[index] -= (
                cholesky_vecs[0:index].T @ cholesky_vecs[0:index, max_error_index]
            )
        cholesky_vecs[index] /= np.sqrt(max_error)
        errors -= cholesky_vecs[index] ** 2
    return cholesky_vecs[:index].reshape((index, n_modes, n_modes))


def low_rank_two_body_decomposition(
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_rank: int | None = None,
    validate: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Low rank decomposition of a two-body tensor.

    The low rank decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::
        h_{pqrs} = \sum_{t} \sum_{k\ell} U^{t}_{pk} U^{t}_{qk} Z^{t}_{k\ell} U^{t}_{r\ell} U^{t}_{s\ell}

    Here each :math:`U^{t}` is a unitary matrix, referred to as a "leaf tensor,"
    and each :math:`Z^{(t)}` is a symmetric matrix, referred to as a "core tensor."

    The number of terms :math:`t` in the decomposition depends on the allowed
    error threshold. A larger error threshold leads to a smaller number of terms.
    Furthermore, the `max_rank` parameter specifies an optional upper bound
    on :math:`t`.

    Note: Currently, only real-valued two-body tensors are supported.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    Args:
        two_body_tensor: The two-body tensor to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_rank: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor.
        validate: Whether to check that the input tensor has the correct symmetries.
            It should be real and symmetric.
        atol: Absolute numerical tolerance for input validation.

    Returns:
        The leaf tensors and the core tensors. Each list of tensors is collected into
        a numpy array, so this method returns a tuple of two numpy arrays.
        Each numpy array will have shape (t, n, n) where t is the rank of the
        decomposition and n is the number of orbitals.

    Raises:
        ValueError: The input tensor does not have the correct symmetries.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    """
    cholesky_vecs = modified_cholesky(
        two_body_tensor,
        error_threshold=error_threshold,
        max_rank=max_rank,
        validate=validate,
        atol=atol,
    )
    n_modes, _, _, _ = two_body_tensor.shape
    leaf_tensors = np.zeros((len(cholesky_vecs), n_modes, n_modes))
    core_tensors = np.zeros((len(cholesky_vecs), n_modes, n_modes))
    for i, mat in enumerate(cholesky_vecs):
        eigs, vecs = np.linalg.eigh(mat)
        leaf_tensors[i] = vecs
        core_tensors[i] = np.outer(eigs, eigs)
    return leaf_tensors, core_tensors
