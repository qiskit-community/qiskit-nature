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

"""Slater determinants."""

from typing import Optional

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from ._helper import _prepare_slater_determinant_jordan_wigner


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
                Please use
                :class:`qiskit_nature.mappers.second_quantization.JordanWignerMapper`
                to construct the qubit mapper.
        """
        if not len(transformation_matrix.shape) == 2:
            raise ValueError("transformation_matrix must be a 2-dimensional array.")

        m, n = transformation_matrix.shape
        if m > n:
            raise ValueError("transformation_matrix must be square.")
        # TODO maybe actually check if the rows are orthonormal

        if qubit_converter is None:
            qubit_converter = QubitConverter(JordanWignerMapper())

        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_converter.mapper, JordanWignerMapper):
            operations = _prepare_slater_determinant_jordan_wigner(register, transformation_matrix)
            for gate, qubits in operations:
                self.append(gate, qubits)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use "
                "qiskit_nature.mappers.second_quantization.JordanWignerMapper "
                "to construct the qubit mapper."
            )
