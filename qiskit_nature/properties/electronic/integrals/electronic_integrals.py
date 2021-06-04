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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import itertools

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

    def __init__(
        self,
        num_body_terms: int,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
    ) -> None:
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
        """
        self._basis = basis
        assert num_body_terms >= 1
        self._num_body_terms = num_body_terms
        self._matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]]
        if basis == ElectronicBasis.SO:
            assert isinstance(matrices, np.ndarray)
            self._matrices = matrices
        else:
            assert len(matrices) == 2 ** num_body_terms
            assert matrices[0] is not None
            self._matrices = matrices

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
