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

"""The QuadraticHamiltonian class."""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg
from qiskit.quantum_info.operators.mixins import TolerancesMixin
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp


class QuadraticHamiltonian(TolerancesMixin):
    r"""A Hamiltonian that is quadratic in the fermionic ladder operators.

    A quadratic Hamiltonian is an operator of the form

    .. math::
        \sum_{p, q} M_{pq} a^\dagger_p a_q
        + \frac12 \sum_{p, q}
            (\Delta_{pq} a^\dagger_p a^\dagger_q + \text{h.c.})
        + \text{constant}

    where :math:`M` is a Hermitian matrix and :math:`\Delta` is an antisymmetric matrix.
    """

    def __init__(
        self,
        hermitian_part: Optional[np.ndarray] = None,
        antisymmetric_part: Optional[np.ndarray] = None,
        constant: float = 0.0,
    ) -> None:
        r"""Initialize a QuadraticHamiltonian.

        Args:
            hermitian_part: The matrix :math:`M` containing the
                coefficients of the terms that conserve particle number.
            antisymmetric_part: The matrix :math:`\Delta` containing the
                coefficients of the terms that do not conserve particle number.
            constant: An additive constant term.

        Raises:
            ValueError: Either a Hermitian part or antisymmetric part must be specified.
        """
        if hermitian_part is not None:
            self._n_orbitals, _ = hermitian_part.shape
        elif antisymmetric_part is not None:
            self._n_orbitals, _ = antisymmetric_part.shape
        else:
            raise ValueError("Either a Hermitian part or antisymmetric part must be specified.")

        # TODO maybe validate matrix properties
        self._hermitian_part = hermitian_part
        self._antisymmetric_part = antisymmetric_part
        self._constant = constant

        if self._hermitian_part is None:
            self._hermitian_part = np.zeros((self._n_orbitals, self._n_orbitals))
        if self._antisymmetric_part is None:
            self._antisymmetric_part = np.zeros((self._n_orbitals, self._n_orbitals))

    @property
    def hermitian_part(self) -> np.ndarray:
        """The matrix of coefficients of terms that conserve particle number."""
        return self._hermitian_part

    @property
    def antisymmetric_part(self) -> np.ndarray:
        """The matrix of coefficients of terms that do not conserve particle number."""
        return self._antisymmetric_part

    @property
    def constant(self) -> float:
        """The constant."""
        return self._constant

    def _fermionic_op(self) -> FermionicOp:
        """Convert to FermionicOp."""
        terms = []
        # TODO I shouldn't have to use string labels
        # diagonal terms
        for i in range(self._n_orbitals):
            terms.append((f"+_{i} -_{i}", self.hermitian_part[i, i]))
        # off-diagonal terms
        for i in range(self._n_orbitals):
            for j in range(i + 1, self._n_orbitals):
                terms.append((f"+_{i} -_{j}", self.hermitian_part[i, j]))
                terms.append((f"+_{j} -_{i}", self.hermitian_part[j, i]))
                terms.append((f"+_{i} +_{j}", self.antisymmetric_part[i, j]))
                terms.append((f"-_{j} -_{i}", self.antisymmetric_part[i, j].conj()))
        # constant
        terms.append(("", self.constant))
        return FermionicOp(terms, register_length=self._n_orbitals)

    def conserves_particle_number(self) -> bool:
        """Whether the Hamiltonian conserves particle number."""
        return np.allclose(self.antisymmetric_part, 0.0)

    def majorana_form(self) -> Tuple[np.ndarray, float]:
        r"""Return the Majorana representation of the Hamiltonian.

        The Majorana representation of a quadratic Hamiltonian is

        .. math::
            \frac{i}{2} \sum_{j, k} A_{jk} f_j f_k + \text{constant}

        where :math:`A` is a real antisymmetric matrix and the :math:`f_i` are
        normalized Majorana fermion operators, which satisfy the relations:

        .. math::
            f_j = \frac{1}{\sqrt{2}} (a^\dagger_j + a_j)

            f_{j + N} = \frac{i}{\sqrt{2}} (a^\dagger_j - a_j)

        Returns:
            - The matrix :math:`A`
            - The constant
        """
        original = np.block(
            [
                [self.antisymmetric_part, self.hermitian_part],
                [-self.hermitian_part.conj(), -self.antisymmetric_part.conj()],
            ]
        )
        eye = np.eye(self._n_orbitals, dtype=complex)
        majorana_basis = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
        matrix = -1j * majorana_basis.conj() @ original @ majorana_basis.T.conj()
        constant = 0.5 * np.trace(self.hermitian_part) + self.constant
        # imaginary parts should be zero
        return np.real(matrix), np.real(constant)

    def diagonalizing_bogoliubov_transform(self) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""Return the transformation matrix that diagonalizes a quadratic Hamiltonian.

        Recall that a quadratic Hamiltonian has the form

        .. math::
            \sum_{p, q} M_{pq} a^\dagger_p a_q
            + \frac12 \sum_{p, q}
                (\Delta_{pq} a^\dagger_p a^\dagger_q + \text{h.c.})
            + \text{constant}

        where the :math:`a^\dagger_j` are fermionic creation operations.
        A quadratic Hamiltonian can always be rewritten in the form

        .. math::
            \sum_{j} \varepsilon_j b^\dagger_j b_j + \text{constant},

        where the :math:`b^\dagger_j` are a new set of fermionic creation operators
        that also satisfy the canonical anticommutation relations.
        These new creation operators are linear combinations of the old ladder
        operators. When the Hamiltonian conserves particle number (:math:`\Delta = 0`)
        then only creation operators need to be mixed together:

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
           \end{pmatrix},

        where :math:`W` is an :math:`N \times N` unitary matrix. However, in the general case,
        both creation and annihilation operators are mixed together:

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

        where now :math:`W` is an :math:`N \times 2N` matrix with orthonormal rows
        (which satisfies additional constraints).

        Returns:
            - The matrix :math:`W`, which is either an :math:`N \times N` or
              an :math:`N \times 2N` matrix
            - A numpy array containing the orbital energies :math:`\varepsilon_j`
              sorted in ascending order
            - The constant
        """
        if self.conserves_particle_number():
            return self._particle_num_conserving_bogoliubov_transform()
        return self._non_particle_num_conserving_bogoliubov_transform()

    def _particle_num_conserving_bogoliubov_transform(self) -> Tuple[np.ndarray, np.ndarray, float]:
        orbital_energies, basis_change = np.linalg.eigh(self.hermitian_part)
        return basis_change.T, orbital_energies, self.constant

    def _non_particle_num_conserving_bogoliubov_transform(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        matrix, constant = self.majorana_form()

        canonical_form, basis_change = _antisymmetric_canonical_form(matrix)
        orbital_energies = canonical_form[
            range(self._n_orbitals), range(self._n_orbitals, 2 * self._n_orbitals)
        ]
        constant -= 0.5 * np.sum(orbital_energies)

        eye = np.eye(self._n_orbitals, dtype=complex)
        majorana_basis = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
        diagonalizing_unitary = majorana_basis.T.conj() @ basis_change @ majorana_basis

        return diagonalizing_unitary[: self._n_orbitals], orbital_energies, constant


def _antisymmetric_canonical_form(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Put an antisymmetric matrix into canonical form.

    The input is an antisymmetric matrix A with even dimension.
    Its canonical form is
    ```
        A = R^T C R
    ```
    where R is a real orthogonal matrix and C has the form
    ```
        C = [  0  D ]
            [ -D  0 ]
    ```
    where D is a real diagonal matrix with non-negative entries
    sorted in ascending order. This form is equivalent to the Schur form
    up to a permutation.

    Returns:
        The matrices C and R.
    """
    dim, _ = matrix.shape
    canonical_form, basis_change = scipy.linalg.schur(matrix, output="real")

    # shift 2x2 blocks so they lie on even indices
    permutation = np.arange(dim)
    for i in range(1, dim - 1, 2):
        if not np.isclose(canonical_form[i + 1, i], 0.0):
            _swap_indices(permutation, i - 1, i + 1)
    _permute_indices(canonical_form, permutation)
    basis_change = basis_change[:, permutation]

    # permute matrix into [[0, A], [-A, 0]] form
    permutation = _schur_permutation(dim)
    _permute_indices(canonical_form, permutation)
    basis_change = basis_change[:, permutation]

    # permute matrix so that the upper right block is non-negative
    n = dim // 2
    permutation = np.arange(dim)
    for i in range(n):
        if canonical_form[i, n + i] < 0.0:
            _swap_indices(permutation, i, n + i)
    _permute_indices(canonical_form, permutation)
    basis_change = basis_change[:, permutation]

    # permute matrix so that the energies are sorted
    permutation = np.argsort(canonical_form[range(n), range(n, 2 * n)])
    permutation = np.concatenate([permutation, permutation + n])
    _permute_indices(canonical_form, permutation)
    basis_change = basis_change[:, permutation]

    return canonical_form, basis_change.T


def _schur_permutation(dim: int) -> np.ndarray:
    n = dim // 2
    permutation = np.arange(dim)
    for i in range(1, n, 2):
        _swap_indices(permutation, i, n + i - 1)
        if n % 2 != 0:
            _swap_indices(permutation, n - 1, n + i)
    return permutation


def _swap_indices(array: np.ndarray, i: int, j: int) -> None:
    array[i], array[j] = array[j], array[i]


def _permute_indices(matrix: np.ndarray, permutation: np.ndarray) -> None:
    n, _ = matrix.shape
    for i in range(n):
        matrix[i, :] = matrix[i, permutation]
    for i in range(n):
        matrix[:, i] = matrix[permutation, i]
