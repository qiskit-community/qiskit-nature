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

"""A base class for raw electronic integrals."""

import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp

from ..bases import ElectronicBasis, ElectronicBasisTransform


class ElectronicIntegrals(ABC):
    """A container for raw electronic integrals.

    This class is a template for ``n``-body electronic integral containers.
    It provides method stubs which must be completed in order to allow basis transformation between
    different ``ElectronicBasis``. An extra method stub must be implemented to map into the special
    ``ElectronicBasis.SO`` basis which is a required intermediate representation of the electronic
    integrals during the process of mapping to a
    ``qiskit_nature.operators.second_quantization.SecondQuantizedOp``.
    """

    INTEGRAL_TRUNCATION_LEVEL = 1e-12

    def __init__(
        self,
        num_body_terms: int,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
        threshold: float = INTEGRAL_TRUNCATION_LEVEL,
    ):
        """
        Args:
            num_body_terms: ``n``, as in the ``n-body`` terms stored in these integrals.
            basis: the basis which these integrals are stored in. If this is initialized with
                ``ElectronicBasis.SO``, these integrals will be used *ad verbatim* during the
                mapping to a ``SecondQuantizedOp``.
            matrices: the matrices (one or many) storing the actual electronic integrals. If this is
                a single matrix, ``basis`` must be set to ``ElectronicBasis.SO``. Refer to the
                documentation of the specific ``n-body`` integral types for the requirements in case
                of multiple matrices.
            threshold: the truncation level below which to treat the integral in the SO matrix as
                zero-valued.

        Raises:
            ValueError: if the number of body terms is less than 1 or if the number of provided
                matrices does not match the number of body term.
            TypeError: if the provided matrix type does not match with the basis or if the first
                matrix is `None`.
        """
        self._validate_num_body_terms(num_body_terms)
        self._validate_matrices(matrices, basis, num_body_terms)
        self._basis = basis
        self._num_body_terms = num_body_terms
        self._matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]] = matrices
        self._threshold = threshold
        self._matrix_representations: List[str] = [""] * len(matrices)

        if basis != ElectronicBasis.SO:
            self._fill_matrices()

    def __repr__(self) -> str:
        string = f"({self._basis.name}) {self._num_body_terms}-Body Terms:\n"
        if self._basis == ElectronicBasis.SO:
            string += self._render_matrix_as_sparse_list(self._matrices)
        else:
            for title, mat in zip(self._matrix_representations, self._matrices):
                string += f"\t{title}\n"
                string += self._render_matrix_as_sparse_list(mat)
        return string

    @staticmethod
    def _render_matrix_as_sparse_list(matrix) -> str:
        string = ""
        nonzero = matrix.nonzero()
        for value, *indices in zip(matrix[nonzero], *nonzero):
            string += f"\t{indices} = {value}\n"
        return string

    @staticmethod
    def _validate_num_body_terms(num_body_terms: int) -> None:
        """Validates the `num_body_terms` setting."""
        if num_body_terms < 1:
            raise ValueError(
                f"The number of body terms must be greater than 0, not '{num_body_terms}'."
            )

    @staticmethod
    def _validate_matrices(
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
        basis: ElectronicBasis,
        num_body_terms: int,
    ) -> None:
        """Validates the `matrices` for a given `basis`."""
        if basis == ElectronicBasis.SO:
            if not isinstance(matrices, np.ndarray):
                raise TypeError(
                    "Initializing integrals in the SO basis requires a single `np.ndarray` for the "
                    f"integrals, not an object of type `{type(matrices)}`."
                )
        else:
            if not isinstance(matrices, tuple):
                raise TypeError(
                    "Initializing integrals in a basis other than SO requires a tuple of "
                    f"`np.ndarray`s for the integrals, not an object of type `{type(matrices)}`."
                )
            if matrices[0] is None:
                raise TypeError("The first matrix in your list of matrices cannot be `None`!")
            if len(matrices) != 2 ** num_body_terms:
                raise ValueError(
                    f"2 to the power of the number of body terms, {2 ** num_body_terms}, does not "
                    f"match the number of provided matrices, {len(matrices)}."
                )

    def _fill_matrices(self) -> None:
        """Fills the internal matrices where `None` placeholders were inserted.

        This method iterates the internal list of matrices and replaces any occurences of `None`
        with the first matrix of the list. In case, more symmetry arguments need to be considered a
        subclass should overwrite this method.
        """
        filled_matrices = []
        for mat in self._matrices:
            if mat is not None:
                filled_matrices.append(mat)
            else:
                filled_matrices.append(self._matrices[0])
        self._matrices = tuple(filled_matrices)

    @abstractmethod
    def transform_basis(self, transform: ElectronicBasisTransform) -> "ElectronicIntegrals":
        """Transforms the integrals according to the given transform object.

        If the integrals are already in the correct basis, ``self`` is returned.

        Args:
            transform: the transformation object with the integral coefficients.

        Returns:
            The transformed ``ElectronicIntegrals``.

        Raises:
            QiskitNatureError: if the integrals do not match
                ``ElectronicBasisTransform.initial_basis``.
        """

    @abstractmethod
    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special ``ElectronicBasis.SO`` basis.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """

    def to_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these electronic integrals.

        This method uses ``to_spin`` internally to map the electronic integrals into the spin
        orbital basis.

        Returns:
            The ``FermionicOp`` given by these electronic integrals.
        """
        spin_matrix = self.to_spin()
        register_length = len(spin_matrix)

        if not np.any(spin_matrix):
            return FermionicOp.zero(register_length)

        return sum(  # type: ignore
            self._create_base_op(indices, spin_matrix[indices], register_length)
            for indices in itertools.product(
                range(register_length), repeat=2 * self._num_body_terms
            )
            if spin_matrix[indices]
        )

    def _create_base_op(self, indices: Tuple[int, ...], coeff: complex, length: int) -> FermionicOp:
        """Creates a single base operator for the given coefficient.

        Args:
            indices: the indices of the current integral.
            coeff: the current integral value.
            length: the register length of the created operator.

        Returns:
            The base operator.
        """
        base_op = FermionicOp(("I_0", coeff), register_length=length)
        for i, op in self._calc_coeffs_with_ops(indices):
            base_op @= FermionicOp(f"{op}_{i}")
        return base_op

    @abstractmethod
    def _calc_coeffs_with_ops(self, indices: Tuple[int, ...]) -> List[Tuple[int, str]]:
        """Maps indices to creation/annihilation operator symbols.

        Args:
            indices: the orbital indices. The length of this tuple must equal ``2 * num_body_term``.

        Returns:
            A list of tuples associating each index with a creation/annihilation operator symbol.
        """

    def add(self, other: "ElectronicIntegrals") -> "ElectronicIntegrals":
        """Adds two ElectronicIntegrals instances.

        Args:
            other: another instance of ElectronicIntegrals.

        Returns:
            The added ElectronicIntegrals.
        """
        ret = deepcopy(self)
        if isinstance(self._matrices, np.ndarray):
            ret._matrices = self._matrices + other._matrices
        else:
            ret._matrices = [a + b for a, b in zip(self._matrices, other._matrices)]
        return ret

    def compose(
        self, other: "ElectronicIntegrals", einsum: Optional[str] = None
    ) -> Union[complex, "ElectronicIntegrals"]:
        """Composes two ElectronicIntegrals instances.

        Args:
            other: another instance of ElectronicIntegrals.
            einsum: an additional `np.einsum` subscript.

        Returns:
            Either a single number or a new instance of ElectronicIntegrals.
        """
        raise NotImplementedError()

    def __rmul__(self, other: complex) -> "ElectronicIntegrals":
        ret = deepcopy(self)
        if isinstance(self._matrices, np.ndarray):
            ret._matrices = other * self._matrices
        else:
            ret._matrices = [other * mat for mat in self._matrices]
        return ret

    def __add__(self, other: "ElectronicIntegrals") -> "ElectronicIntegrals":
        if self._basis != other._basis:
            raise TypeError()
        return self.add(other)

    def __sub__(self, other: "ElectronicIntegrals") -> "ElectronicIntegrals":
        return self + (-1.0) * other
