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

from qiskit import QuantumRegister, QuantumCircuit

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from ._helper import _prepare_fermionic_gaussian_state_jordan_wigner


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
                This matrix must satisfy special constraints; see the main body of the docstring
                of this function.
            occupied_orbitals: The pseudo-particle orbitals to fill. These refer to the indices
                of the operators :math:`\{b^\dagger_j\}` from the main body of the docstring
                of this function. The default behavior is to use the empty set of orbitals,
                which corresponds to a state with zero pseudo-particles.
            qubit_converter: a QubitConverter instance.
            circuit_kwargs: Keyword arguments to pass to the QuantumCircuit initializer.

        Raises:
            ValueError: transformation_matrix must be a 2-dimensional array.
            ValueError: transformation_matrix must have shape (n_orbitals, 2 * n_orbitals).
            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
                Please use
                :class:`qiskit_nature.mappers.second_quantization.JordanWignerMapper`
                to construct the qubit mapper.
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

        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            operations = _prepare_fermionic_gaussian_state_jordan_wigner(
                register, transformation_matrix, occupied_orbitals
            )
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use "
                "qiskit_nature.mappers.second_quantization.JordanWignerMapper "
                "to construct the qubit mapper."
            )
