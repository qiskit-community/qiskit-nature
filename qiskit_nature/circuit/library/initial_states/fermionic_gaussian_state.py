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

"""Fermionic Gaussian states."""

from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from .utils.givens_rotations import _prepare_fermionic_gaussian_state_jw


def _validate_transformation_matrix(
    mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    if not len(mat.shape) == 2:
        raise ValueError(
            "transformation_matrix must be a 2-dimensional array. "
            f"Instead, got shape {mat.shape}."
        )

    n, p = mat.shape  # pylint: disable=invalid-name
    if p != n * 2:
        raise ValueError(
            "transformation_matrix must have shape (n_orbitals, 2 * n_orbitals). "
            f"Instead, got shape {mat.shape}."
        )

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


class FermionicGaussianState(QuantumCircuit):
    r"""A circuit that prepares a fermionic Gaussian state.

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

    The matrix :math:`W` is commonly obtained by calling the
    :meth:`~.QuadraticHamiltonian.diagonalizing_bogoliubov_transform`
    method of the :class:`~.QuadraticHamiltonian` class.
    This matrix is used to create circuits that prepare eigenstates of the
    quadratic Hamiltonian.

    Currently, only the Jordan-Wigner Transformation is supported.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395
    """

    def __init__(
        self,
        transformation_matrix: np.ndarray,
        occupied_orbitals: Optional[Sequence[int]] = None,
        qubit_converter: QubitConverter = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **circuit_kwargs,
    ) -> None:
        r"""
        Args:
            transformation_matrix: The matrix :math:`W` that specifies the coefficients of the
                new creation operators in terms of the original creation and annihilation operators.
                This matrix must satisfy special constraints, as detailed above.
            occupied_orbitals: The pseudo-particle orbitals to fill. These refer to the indices
                of the operators :math:`\{b^\dagger_j\}` from the main body of the docstring
                of this function. The default behavior is to use the empty set of orbitals,
                which corresponds to a state with zero pseudo-particles.
            qubit_converter: The qubit converter. The default behavior is to create
                one using the call `QubitConverter(JordanWignerMapper())`.
            validate: Whether to validate the inputs.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.
            circuit_kwargs: Keyword arguments to pass to the QuantumCircuit initializer.

        Raises:
            ValueError: transformation_matrix must be a 2-dimensional array.
            ValueError: transformation_matrix must have shape (n_orbitals, 2 * n_orbitals).
            ValueError: transformation_matrix does not describe a valid transformation
                of fermionic ladder operators. A valid matrix has the block form
                :math:`(W_1 \quad W_2)` where :math:`W_1 W_1^\dagger + W_2 W_2^\dagger = I` and
                :math:`W_1 W_2^T + W_2 W_1^T = 0`.

            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
                Please use
                :class:`qiskit_nature.mappers.second_quantization.JordanWignerMapper`
                to construct the qubit mapper used to construct `qubit_converter`.
        """
        if validate:
            _validate_transformation_matrix(transformation_matrix, rtol=rtol, atol=atol)

        if occupied_orbitals is None:
            occupied_orbitals = []
        if qubit_converter is None:
            qubit_converter = QubitConverter(JordanWignerMapper())

        n, _ = transformation_matrix.shape
        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            operations = _prepare_fermionic_gaussian_state_jw(
                register, transformation_matrix, occupied_orbitals
            )
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use "
                "qiskit_nature.mappers.second_quantization.JordanWignerMapper "
                "to construct the qubit mapper used to construct qubit_converter."
            )
