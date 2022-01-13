# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic Gaussian states."""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import RZGate, XGate, XYGate

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper


class SlaterDeterminant(QuantumCircuit):
    """A circuit that prepares a Slater determinant."""

    def __init__(
        self,
        transformation_matrix: np.ndarray,
        qubit_converter: Optional[QubitConverter] = None,
        **circuit_kwargs,
    ) -> None:
        r"""Initialize a circuit that prepares a Slater determinant.

        A Slater determinant is a state of the form

        .. math::
            b^\dagger_1 \cdots b^\dagger_{N_f} \lvert \text{vac} \rangle

        where

        .. math::
            b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k,

        - :math:`Q` is an :math:`N_f \times N` matrix with orthonormal rows
        - :math:`a^\dagger_1, \ldots, a^\dagger_{N}` are the fermionic creation operators
        - :math:`\lvert \text{vac} \rangle` is the vacuum state
          (mutual 0-eigenvector of the fermionic number operators :math:`\{a^\dagger_j a_j\}`)

        Currently, only the Jordan-Wigner Transformation is supported.

        Reference: arXiv:1711.05395

        Args:
            transformation_matrix: The matrix :math:`Q` that specifies the coefficients of the
                new creation operators in terms of the original creation operators.
                The rows of the matrix must be orthonormal.
            qubit_converter: A QubitConverter instance.
            circuit_kwargs: Keyword arguments to pass to the QuantumCircuit initializer.

        Raises:
            ValueError: transformation_matrix must be a 2-dimensional array.
            ValueError: transformation_matrix must have orthonormal rows.
            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
        """
        if not len(transformation_matrix.shape) == 2:
            raise ValueError("transformation_matrix must be a 2-dimensional array.")

        m, n = transformation_matrix.shape
        if m > n:
            raise ValueError("transformation_matrix must have orthonormal rows.")
        # TODO maybe actually check if the rows are orthonormal

        if qubit_converter is None:
            qubit_converter = QubitConverter(JordanWignerMapper())

        register = QuantumRegister(n, "q")
        # TODO maybe use a shorter name
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            operations = _prepare_slater_determinant_jordan_wigner(register, transformation_matrix)
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError("Currently, only the Jordan-Wigner Transform is supported.")


class FermionicGaussianState(QuantumCircuit):
    """A circuit that prepares a fermionic Gaussian state."""

    def __init__(
        self,
        transformation_matrix: np.ndarray,
        occupied_orbitals: Optional[Sequence[int]] = None,
        qubit_converter: QubitConverter = None,
        **circuit_kwargs,
    ) -> None:
        r"""Initialize a circuit that prepares a fermionic Gaussian state.

        A fermionic Gaussian state is a state of the form

        .. math::
            b^\dagger_1 \cdots b^\dagger_{N_p} \lvert \overline{\text{vac}} \rangle

        where

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
           \end{pmatrix},

        - :math:`a^\dagger_1, \ldots, a^\dagger_{N}` are the fermionic creation operators
        - :math:`W` is an :math:`N \times 2N` matrix such that
          :math:`b^\dagger_1, \ldots, b^\dagger_{N}` also satisfy the
          fermionic anticommutation relations
        - :math:`\lvert \overline{\text{vac}} \rangle` is the mutual 0-eigenvector of
          the operators :math:`\{b_j^\dagger b_j\}`

        The matrix :math:`W` has the block form

        .. math::
           \begin{pmatrix}
                W_1 & W_2
           \end{pmatrix}

        where :math:`W_1` and :math:`W_2` must satisfy

        .. math::
            W_1 W_1^\dagger + W_2 W_2^\dagger = I \\
            W_1 W_2^T + W_2 W_1^T = 0

        Currently, only the Jordan-Wigner Transformation is supported.

        Reference: arXiv:1711.05395

        Args:
            transformation_matrix: The matrix :math:`W` that specifies the coefficients of the
                new creation operators in terms of the original creation and annihilation operators.
                This matrix must satisfy special constraints; see the docstring of this function.
            qubit_converter: a QubitConverter instance.
            circuit_kwargs: Keyword arguments to pass to the QuantumCircuit initializer.

        Raises:
            ValueError: transformation_matrix must be a 2-dimensional array.
            ValueError: transformation_matrix must have shape (n_orbitals, 2 * n_orbitals).
            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
        """
        if not len(transformation_matrix.shape) == 2:
            raise ValueError("transformation_matrix must be a 2-dimensional array.")

        n, p = transformation_matrix.shape  # pylint: disable=invalid-name
        if p != n * 2:
            raise ValueError("transformation_matrix must have shape (n_orbitals, 2 * n_orbitals).")
        # TODO maybe check known matrix constraints

        if occupied_orbitals is None:
            occupied_orbitals = []
        if qubit_converter is None:
            qubit_converter = QubitConverter(JordanWignerMapper())

        register = QuantumRegister(n, "q")
        # TODO maybe use a shorter name
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            operations = _prepare_fermionic_gaussian_state_jordan_wigner(
                register, transformation_matrix, occupied_orbitals
            )
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError("Currently, only the Jordan-Wigner Transform is supported.")


def _prepare_slater_determinant_jordan_wigner(  # pylint: disable=invalid-name
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
                    (XYGate(2 * theta, -np.pi / 2), (register[j], register[j - 1]))
                )
                # update matrix
                current_matrix = _apply_matrix_to_slices(
                    current_matrix, givens_matrix, [(Ellipsis, j), (Ellipsis, j - 1)]
                )

    yield from reversed(decomposition)


def _prepare_fermionic_gaussian_state_jordan_wigner(  # pylint: disable=invalid-name
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
                        (XYGate(2 * theta, -np.pi / 2), (register[j], register[j + 1]))
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

    yield from _prepare_slater_determinant_jordan_wigner(
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
