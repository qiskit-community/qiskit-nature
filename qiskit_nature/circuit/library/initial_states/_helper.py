# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Private helper functions for inital states."""

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import RZGate, XGate, XXPlusYYGate


def prepare_slater_determinant_jordan_wigner(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    m, n = transformation_matrix.shape

    # set the first n_particles qubits to 1
    for i in range(m):
        yield XGate(), (register[i],)

    # if all orbitals are filled, no further operations are needed
    if m == n:
        return

    current_matrix = transformation_matrix

    # zero out top right corner by rotating rows; this is a no-op
    for j in reversed(range(n - m + 1, n)):
        # Zero out entries in column j
        for i in range(m - n + j):
            # Zero out entry in row i if needed
            if not np.isclose(current_matrix[i, j], 0.0):
                givens_matrix = _givens_matrix(current_matrix[i, j], current_matrix[i + 1, j])
                current_matrix = _apply_matrix_to_slices(current_matrix, givens_matrix, [i, i + 1])

    # decompose matrix into Givens rotations
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    for i in range(m):
        # zero out the columns in row i
        for j in range(n - m + i, i, -1):
            if not np.isclose(current_matrix[i, j], 0.0):
                # compute Givens rotation
                givens_matrix = _givens_matrix(current_matrix[i, j], current_matrix[i, j - 1])
                theta = np.arcsin(np.real(givens_matrix[1, 0]))
                phi = -np.angle(givens_matrix[1, 1])
                # add operations
                decomposition.append((RZGate(phi), (register[j - 1],)))
                decomposition.append(
                    (XXPlusYYGate(2 * theta, -np.pi / 2), (register[j], register[j - 1]))
                )
                # update matrix
                current_matrix = _apply_matrix_to_slices(
                    current_matrix, givens_matrix, [(Ellipsis, j), (Ellipsis, j - 1)]
                )

    yield from reversed(decomposition)


def prepare_fermionic_gaussian_state_jordan_wigner(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray, occupied_orbitals: Sequence[int]
) -> Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
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
                givens_matrix = _givens_matrix(current_matrix[i, j], current_matrix[i + 1, j])
                current_matrix = _apply_matrix_to_slices(current_matrix, givens_matrix, [i, i + 1])
                left_unitary = _apply_matrix_to_slices(left_unitary, givens_matrix, [i, i + 1])

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
                    givens_matrix = _givens_matrix(current_matrix[i, j], current_matrix[i, j + 1])
                    theta = np.arcsin(np.real(givens_matrix[1, 0]))
                    phi = -np.angle(givens_matrix[1, 1])
                    # add operations
                    decomposition.append((RZGate(phi), (register[j + 1],)))
                    decomposition.append(
                        (XXPlusYYGate(2 * theta, -np.pi / 2), (register[j], register[j + 1]))
                    )
                    # update matrix
                    current_matrix = _apply_matrix_to_slices(
                        current_matrix,
                        givens_matrix,
                        [(Ellipsis, j), (Ellipsis, j + 1)],
                    )
                    current_matrix = _apply_matrix_to_slices(
                        current_matrix,
                        givens_matrix.conj(),
                        [(Ellipsis, n + j), (Ellipsis, n + j + 1)],
                    )

    for i in range(n):
        left_unitary[i] *= current_matrix[i, n + i].conj()

    yield from prepare_slater_determinant_jordan_wigner(
        register, left_unitary.T[list(occupied_orbitals)]
    )
    yield from reversed(decomposition)


def _givens_matrix(a: Union[complex, float], b: Union[complex, float]) -> np.ndarray:
    """Compute the Givens rotation to zero out a row entry.

    Returns a 2x2 unitary matrix G that satisfies
    ```
    G * [a  b]^T= [0  r]^T
    ```
    where `r` is a complex number.

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
        phase = 1.0
    # Handle case that b is zero and a is nonzero
    elif np.isclose(b, 0.0):
        cosine = 0.0
        sine = 1.0
        phase = 1.0
    # Handle case that a and b are both nonzero
    else:
        denominator = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        cosine = abs(b) / denominator
        sine = abs(a) / denominator
        sign_b = b / abs(b)
        sign_a = a / abs(a)
        phase = sign_a * sign_b.conjugate()
        # If phase is a real number, convert it to a float
        if np.isreal(phase):
            phase = np.real(phase)

    return np.array([[cosine, -phase * sine], [sine, phase * cosine]])


_SliceAtom = Union[int, slice, "ellipsis"]
_Slice = Union[_SliceAtom, Sequence[_SliceAtom]]


def _apply_matrix_to_slices(
    target: np.ndarray, matrix: np.ndarray, slices: Sequence[_Slice]
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor."""
    result = target.copy()
    for i, slice_i in enumerate(slices):
        result[slice_i] *= matrix[i, i]
        for j, slice_j in enumerate(slices):
            if j != i:
                result[slice_i] += target[slice_j] * matrix[i, j]
    return result


def _swap_columns(matrix: np.ndarray, i: int, j: int) -> None:
    """Swap columns of a matrix, mutating it."""
    column_i = matrix[:, i].copy()
    column_j = matrix[:, j].copy()
    matrix[:, i], matrix[:, j] = column_j, column_i
