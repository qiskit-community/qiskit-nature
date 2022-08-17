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

"""A base class for raw electronic integrals."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generator, Optional, Union

import h5py
import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp

from ..bases import ElectronicBasis, ElectronicBasisTransform
from .....deprecation import warn_deprecated, DeprecatedType, NatureDeprecationWarning


class ElectronicIntegrals(ABC):
    """A container for raw electronic integrals.

    This class is a template for ``n``-body electronic integral containers.
    It provides method stubs which must be completed in order to allow basis transformation between
    different
    :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis`. An
    extra method stub must be implemented to map into the special
    :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO` basis
    which is a required intermediate representation of the electronic integrals during the process
    of mapping to a :class:`~qiskit_nature.operators.second_quantization.SecondQuantizedOp`.

    When these integrals are printed the output will be truncated based on the
    ``ElectronicIntegrals._truncate`` value (defaults to 5). Use
    ``ElectronicIntegrals.set_truncation`` to change this value.
    """

    VERSION = 1

    INTEGRAL_TRUNCATION_LEVEL = 1e-12

    _MATRIX_REPRESENTATIONS: list[str] = []

    _truncate = 5

    def __init__(
        self,
        num_body_terms: int,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, tuple[Optional[np.ndarray], ...]],
        threshold: float = INTEGRAL_TRUNCATION_LEVEL,
    ) -> None:
        # pylint: disable=line-too-long
        """
        Args:
            num_body_terms: ``n``, as in the ``n-body`` terms stored in these integrals.
            basis: the basis which these integrals are stored in. If this is initialized with
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`,
                these integrals will be used *ad verbatim* during the mapping to a
                :class:`~qiskit_nature.operators.second_quantization.SecondQuantizedOp`.
            matrices: the matrices (one or many) storing the actual electronic integrals. If this is
                a single matrix, ``basis`` must be set to
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`.
                Refer to the documentation of the specific ``n-body`` integral types for the
                requirements in case of multiple matrices.
            threshold: the truncation level below which to treat the integral as zero-valued.

        Raises:
            ValueError: if the number of body terms is less than 1 or if the number of provided
                matrices does not match the number of body term.
            TypeError: if the provided matrix type does not match with the basis or if the first
                matrix is ``None``.
        """
        warn_deprecated(
            "0.5.0",
            old_type=DeprecatedType.CLASS,
            old_name="qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals",
            new_type=DeprecatedType.CLASS,
            new_name="qiskit_nature.second_q.operators.PolynomialTensor",
            category=NatureDeprecationWarning,
        )
        self._validate_num_body_terms(num_body_terms)
        self._validate_matrices(matrices, basis, num_body_terms)
        self.name = self.__class__.__name__
        self._basis = basis
        self._num_body_terms = num_body_terms
        self._threshold = threshold
        self._matrices: Union[np.ndarray, tuple[Optional[np.ndarray], ...]]
        if basis == ElectronicBasis.SO:
            self._matrices = np.where(np.abs(matrices) > self._threshold, matrices, 0.0)
        else:
            self._matrices = tuple(
                np.where(np.abs(mat) > self._threshold, mat, 0.0) if mat is not None else None
                for mat in matrices
            )

    def get_matrix(self, index: int = 0) -> np.ndarray:
        """Returns the integral matrix at the requested index.

        When the integrals are in the :class:`ElectronicBasis.SO` basis the index has no effect
        because only a single matrix exists.
        In any other case, the index may not exceed the length of the internal matrices being
        stored. This length depends on the number of body terms for the specific implementation of
        this base class.

        Finally, when the internal matrix encountered at the requested index is `None`, it will be
        replaced by the matrix at index 0, which for most common purposes is the default fallback
        (pure alpha-spin) integral matrix used in place of other mixed spin matrices.
        If a subclass must follow another convention it should overwrite the implementation of this
        method.

        Args:
            index: the index of the integral matrix to get.

        Returns:
            The requested integral matrix.

        Raises:
            IndexError: when the requested index exceeds the number of internal matrices.
        """
        if isinstance(self._matrices, np.ndarray):
            return self._matrices

        if index >= len(self._matrices):
            raise IndexError(
                f"The requested index {index} exceeds the number of internal matrices "
                f"{len(self._matrices)}."
            )

        mat = self._matrices[index]
        if mat is None:
            mat = self._matrices[0]

        return mat

    @property
    def basis(self) -> ElectronicBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: ElectronicBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @property
    def num_body_terms(self) -> int:
        """Returns the num_body_terms."""
        return self._num_body_terms

    @num_body_terms.setter
    def num_body_terms(self, num_body_terms: int) -> None:
        """Sets the num_body_terms."""
        self._num_body_terms = num_body_terms

    @property
    def threshold(self) -> float:
        """Returns the threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """Sets the threshold."""
        self._threshold = threshold

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        group = parent.require_group(self.name)
        group.attrs["__class__"] = self.__class__.__name__
        group.attrs["__module__"] = self.__class__.__module__
        group.attrs["__version__"] = self.VERSION

        group.attrs["basis"] = self._basis.name
        group.attrs["threshold"] = self._threshold

        if self._basis == ElectronicBasis.SO:
            group.create_dataset("Spin", data=self._matrices)
        else:
            for name, mat in zip(self._MATRIX_REPRESENTATIONS, self._matrices):
                if mat is None:
                    continue
                group.create_dataset(name, data=mat)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> ElectronicIntegrals:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        class_name = h5py_group.attrs["__class__"]
        module_path = h5py_group.attrs["__module__"]

        loaded_module = importlib.import_module(module_path)
        loaded_class = getattr(loaded_module, class_name, None)

        basis = getattr(ElectronicBasis, h5py_group.attrs["basis"])
        threshold = h5py_group.attrs["threshold"]
        matrices: list[Optional[np.ndarray]] = []
        for rep in loaded_class._MATRIX_REPRESENTATIONS:
            if rep in h5py_group.keys():
                matrices.append(h5py_group[rep][...])
            else:
                matrices.append(None)

        return loaded_class(
            basis=basis,
            matrices=tuple(matrices),
            threshold=threshold,
        )

    def __str__(self) -> str:
        string = [f"({self._basis.name}) {self._num_body_terms}-Body Terms:"]
        if self._basis == ElectronicBasis.SO:
            string += self._render_matrix_as_sparse_list(self._matrices)
        else:
            for title, mat in zip(self._MATRIX_REPRESENTATIONS, self._matrices):
                string += [f"\t{title}"]
                if mat is None:
                    string += ["\tSame values as the corresponding primary-spin data."]
                    continue
                rendered_matrix = self._render_matrix_as_sparse_list(mat)
                if not rendered_matrix:
                    string[-1] += " is all zero"
                    continue
                string += rendered_matrix
        return "\n".join(string)

    @staticmethod
    def _render_matrix_as_sparse_list(matrix) -> list[str]:
        string = []
        nonzero = matrix.nonzero()
        nonzero_count = len(nonzero[0])
        string += [f"\t<{matrix.shape} matrix with {nonzero_count} non-zero entries>"]
        count = 0
        for value, *indices in zip(matrix[nonzero], *nonzero):
            if ElectronicIntegrals._truncate and count >= ElectronicIntegrals._truncate:
                string += [
                    f"\t... skipping {nonzero_count - ElectronicIntegrals._truncate} entries"
                ]
                break
            string += [f"\t{indices} = {value}"]
            count += 1
        return string

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """An iterator over the internal matrices."""
        if isinstance(self._matrices, np.ndarray):
            yield self._matrices
        else:
            for mat in self._matrices:
                yield mat

    @staticmethod
    def set_truncation(max_num_entries: int) -> None:
        """Set the maximum number of integral values to display before truncation.

        Args:
            max_num_entries: the maximum number of entries.

        .. note::
            Truncation will be disabled if `max_num_entries` is set to 0.
        """
        ElectronicIntegrals._truncate = max_num_entries

    @staticmethod
    def _validate_num_body_terms(num_body_terms: int) -> None:
        """Validates the number of body terms."""
        if num_body_terms < 1:
            raise ValueError(
                f"The number of body terms must be greater than 0, not '{num_body_terms}'."
            )

    @staticmethod
    def _validate_matrices(
        matrices: Union[np.ndarray, tuple[Optional[np.ndarray], ...]],
        basis: ElectronicBasis,
        num_body_terms: int,
    ) -> None:
        """Validates the ``matrices`` for a given ``basis``."""
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
            if len(matrices) != 2**num_body_terms:
                raise ValueError(
                    f"2 to the power of the number of body terms, {2 ** num_body_terms}, does not "
                    f"match the number of provided matrices, {len(matrices)}."
                )

    @abstractmethod
    def transform_basis(self, transform: ElectronicBasisTransform) -> ElectronicIntegrals:
        # pylint: disable=line-too-long
        """Transforms the integrals according to the given transform object.

        If the integrals are already in the correct basis, ``self`` is returned.

        Args:
            transform: the transformation object with the integral coefficients.

        Returns:
            The transformed
            :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals`.

        Raises:
            QiskitNatureError: if the integrals do not match
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasisTransform.initial_basis`.
        """

    @abstractmethod
    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special
        :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`
        basis.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """

    def to_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these electronic integrals.

        This method uses ``to_spin`` internally to map the electronic integrals into the spin
        orbital basis.

        Returns:
            The :class:`~qiskit_nature.operators.second_quantization.FermionicOp` given by these
            electronic integrals.
        """
        spin_matrix = self.to_spin()
        register_length = len(spin_matrix)

        if not np.any(spin_matrix):
            return FermionicOp.zero(register_length)

        spin_matrix_iter = spin_matrix.flat
        # NOTE: we need to access `.coords` before `.next()` is called for the first time!
        coords = spin_matrix_iter.coords
        op_data = []
        for coeff in spin_matrix_iter:
            if coeff:
                op_data.append((self._calc_coeffs_with_ops(coords), coeff))
            coords = spin_matrix_iter.coords

        return FermionicOp(op_data, register_length=register_length, display_format="sparse")

    @staticmethod
    @abstractmethod
    def _calc_coeffs_with_ops(indices: tuple[int, ...]) -> list[tuple[str, int]]:
        """Maps indices to creation/annihilation operator symbols.

        Args:
            indices: the orbital indices. The length of this tuple must equal ``2 * num_body_term``.

        Returns:
            A list of tuples associating each index with a creation/annihilation operator symbol.
        """

    def add(self, other: ElectronicIntegrals) -> ElectronicIntegrals:
        """Adds two ElectronicIntegrals instances.

        Args:
            other: another instance of ElectronicIntegrals.

        Returns:
            The added ElectronicIntegrals.
        """
        ret = deepcopy(self)
        if self._basis == ElectronicBasis.SO:
            ret._matrices = self._matrices + other._matrices
        else:
            ret_matrices: list[Optional[np.ndarray]] = []
            for idx, (mat_a, mat_b) in enumerate(zip(self._matrices, other._matrices)):
                if mat_a is None and mat_b is None:
                    ret_matrices.append(None)
                    continue
                if mat_a is None:
                    mat_a = self.get_matrix(idx)
                if mat_b is None:
                    mat_b = other.get_matrix(idx)
                ret_matrices.append(mat_a + mat_b)
            ret._matrices = tuple(ret_matrices)
        return ret

    def compose(
        self, other: ElectronicIntegrals, einsum_subscript: Optional[str] = None
    ) -> Union[complex, ElectronicIntegrals]:
        """Composes two ``ElectronicIntegrals`` instances.

        Args:
            other: another instance of ``ElectronicIntegrals``.
            einsum_subscript: an additional ``np.einsum`` subscript.

        Returns:
            Either a single number or a new instance of ``ElectronicIntegrals``.
        """
        raise NotImplementedError()

    def __rmul__(self, other: complex) -> ElectronicIntegrals:
        ret = deepcopy(self)
        if isinstance(self._matrices, np.ndarray):
            ret._matrices = other * self._matrices
        else:
            ret._matrices = tuple(None if mat is None else other * mat for mat in self._matrices)
        return ret

    def __add__(self, other: ElectronicIntegrals) -> ElectronicIntegrals:
        if self._basis != other._basis:
            raise ValueError(
                f"The basis of self, {self._basis.value}, does not match the basis of other, "
                f"{other._basis}!"
            )
        return self.add(other)

    def __sub__(self, other: ElectronicIntegrals) -> ElectronicIntegrals:
        return self + (-1.0) * other
