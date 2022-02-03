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

"""A container for arbitrary ``n-body`` vibrational integrals."""

from __future__ import annotations

from abc import ABC
from collections import Counter
from itertools import chain, cycle, permutations, product, tee
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import VibrationalOp

from ..bases import VibrationalBasis


class VibrationalIntegrals(ABC):
    """A container for arbitrary ``n-body`` vibrational integrals.

    When these integrals are printed the output will be truncated based on the
    ``VibrationalIntegrals._truncate`` value (defaults to 5). Use
    ``VibrationalIntegrals.set_truncation`` to change this value.
    """

    VERSION = 1

    _truncate = 5

    def __init__(
        self,
        num_body_terms: int,
        integrals: List[Tuple[float, Tuple[int, ...]]],
    ) -> None:
        """
        Args:
            num_body_terms: ``n``, as in the ``n-body`` terms stored in these integrals.
            integrals: a sparse list of integrals. The data format corresponds to a list of pairs,
                with its first entry being the integral coefficient and the second entry being a
                tuple of integers of length ``num_body_terms``. These integers are the indices of
                the modes associated with the integral. If the indices are negative, the integral is
                treated as a kinetic term of the vibrational hamiltonian.

        Raises:
            ValueError: if the number of body terms is less than 1.
        """
        self._validate_num_body_terms(num_body_terms)
        self.name = f"{num_body_terms}Body{self.__class__.__name__}"
        self._num_body_terms = num_body_terms
        self._integrals = integrals
        self._basis: VibrationalBasis = None

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @property
    def integrals(self) -> List[Tuple[float, Tuple[int, ...]]]:
        """Returns the integrals."""
        return self._integrals

    @integrals.setter
    def integrals(self, integrals: List[Tuple[float, Tuple[int, ...]]]) -> None:
        """Sets the integrals."""
        self._integrals = integrals

    def __str__(self) -> str:
        string = [f"{self._num_body_terms}-Body Terms:"]
        integral_count = len(self._integrals)
        string += [f"\t\t<sparse integral list with {integral_count} entries>"]
        count = 0
        for value, indices in self._integrals:
            if VibrationalIntegrals._truncate and count >= VibrationalIntegrals._truncate:
                string += [
                    f"\t\t... skipping {integral_count - VibrationalIntegrals._truncate} entries"
                ]
                break
            string += [f"\t\t{indices} = {value}"]
            count += 1
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :class:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        group = parent.require_group(self.name)
        group.attrs["__class__"] = self.__class__.__name__
        group.attrs["__module__"] = self.__class__.__module__
        group.attrs["__version__"] = self.VERSION

        group.attrs["num_body_terms"] = self._num_body_terms

        dtype = h5py.vlen_dtype(np.dtype("int32"))
        integrals_dset = group.create_dataset("integrals", (len(self.integrals),), dtype=dtype)
        coeffs_dset = group.create_dataset("coefficients", (len(self.integrals),), dtype=float)

        for idx, ints in enumerate(self.integrals):
            coeffs_dset[idx] = ints[0]
            integrals_dset[idx] = list(ints[1])

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> VibrationalIntegrals:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :class:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        integrals = []
        for coeff, ints in zip(h5py_group["coefficients"][...], h5py_group["integrals"][...]):
            integrals.append((coeff, tuple(ints)))

        ret = VibrationalIntegrals(h5py_group.attrs["num_body_terms"], integrals)

        return ret

    @staticmethod
    def set_truncation(max_num_entries: int) -> None:
        """Set the maximum number of integral values to display before truncation.

        Args:
            max_num_entries: the maximum number of entries.

        .. note::
            Truncation will be disabled if `max_num_entries` is set to 0.
        """
        VibrationalIntegrals._truncate = max_num_entries

    @staticmethod
    def _validate_num_body_terms(num_body_terms: int) -> None:
        """Validates the number of body terms."""
        if num_body_terms < 1:
            raise ValueError(
                f"The number of body terms must be greater than 0, not '{num_body_terms}'."
            )

    def to_basis(self) -> np.ndarray:
        """Maps the integrals into a basis which permits mapping into second-quantization.

        Returns:
            A single matrix containing the ``n-body`` integrals in the mapped basis.

        Raises:
            QiskitNatureError: if no basis has been set yet.
            ValueError: if a mismatching integral set and number of body terms is encountered.
        """
        if self._basis is None:
            raise QiskitNatureError("You must set a basis first!")

        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros((num_modes, max_num_modals, max_num_modals) * self._num_body_terms)

        # we can cache already evaluated integrals to improve cases in which a basis is very
        # expensive to compute
        coeff_cache: Dict[Tuple[int, int, int, int, bool], Optional[float]] = {}

        for coeff0, indices in self._integrals:
            if len(set(indices)) != self._num_body_terms:
                raise ValueError(
                    f"The number of body terms, {self._num_body_terms}, does not match the number "
                    f"of different indices in your integral, {len(set(indices))}."
                )

            indices_np = np.asarray(indices, dtype=int)
            # NOTE: negative indices may be treated specially by a basis
            kinetic_term = any(index < 0 for index in indices_np)
            if kinetic_term:
                # once we have determined whether a term is kinetic, all indices must be positive
                indices_np = np.absolute(indices_np)

            # the number of times which an index occurs corresponds to the power of the operator
            powers: Dict[int, int] = Counter(indices_np)

            index_list = []

            # we do an initial loop to evaluate all relevant basis integrals
            for mode, power in powers.items():
                iter_1, iter_2 = tee(zip(*np.tril_indices(num_modals_per_mode[mode - 1])))
                # we must store the indices of the mode in combination with all possible modal
                # permutations (lower triangular indices) for the next step
                index_list.append(zip(cycle([mode]), iter_1))
                for m, n in iter_2:
                    if (mode - 1, m, n, power, kinetic_term) in coeff_cache:
                        # value already in cache
                        continue
                    coeff_cache[(mode - 1, m, n, power, kinetic_term)] = self.basis.eval_integral(
                        mode - 1, m, n, power, kinetic_term=kinetic_term
                    )

            # now we can iterate the product of all index lists (the cartesian product is equivalent
            # to nested for loops but has the benefit of being agnostic w.r.t. the number of body
            # terms)
            for index in product(*index_list):
                index_permutations = []
                coeff = coeff0
                for mode, (m, n) in index:
                    # compute the total coefficient
                    cached_coeff = coeff_cache[(mode - 1, m, n, powers[mode], kinetic_term)]
                    if cached_coeff is None:
                        break
                    coeff *= cached_coeff
                    index_set = set()
                    # generate potentially symmetric permutations of the modal indices
                    for m_sub, n_sub in permutations((m, n)):
                        index_set.add((m_sub, n_sub))
                    index_permutations.append(
                        {(mode - 1, m_sub, n_sub) for (m_sub, n_sub) in index_set}
                    )
                else:
                    # update the matrix in all permuted locations
                    for i in product(*index_permutations):
                        matrix[tuple(chain(*i))] += coeff

        return matrix

    def to_second_q_op(self) -> VibrationalOp:
        """Creates the operator representing the Hamiltonian defined by these vibrational integrals.

        Returns:
            The :class:`~qiskit_nature.operators.second_quantization.VibrationalOp` given by these
            vibrational integrals.

        Raises:
            QiskitNatureError: if no basis has been set yet.
        """
        try:
            matrix = self.to_basis()
        except QiskitNatureError as exc:
            raise QiskitNatureError() from exc

        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        nonzero = np.nonzero(matrix)

        if not np.any(np.asarray(nonzero)):
            return VibrationalOp.zero(num_modes, num_modals_per_mode)

        labels = []

        for coeff, indices in zip(matrix[nonzero], zip(*nonzero)):
            # the indices need to be grouped into triplets of the form: (mode, modal_1, modal_2)
            grouped_indices = [
                tuple(int(j) for j in indices[i : i + 3]) for i in range(0, len(indices), 3)
            ]
            # the index groups need to processed in sorted order to produce a valid label
            coeff_label = self._create_label_for_coeff(sorted(grouped_indices))
            labels.append((coeff_label, coeff))

        return VibrationalOp(labels, num_modes, num_modals_per_mode)

    @staticmethod
    def _create_label_for_coeff(indices: List[Tuple[int, ...]]) -> str:
        """Generates the operator label for the given indices.

        Args:
            indices: A list of index triplets, where the first number is the mode index and the
                second and third numbers are the modal indices of that mode.

        Returns:
            The constructed operator label.
        """
        complete_labels_list = []
        for mode, modal_raise, modal_lower in indices:
            if modal_raise <= modal_lower:
                complete_labels_list.append(f"+_{mode}*{modal_raise}")
                complete_labels_list.append(f"-_{mode}*{modal_lower}")
            else:
                complete_labels_list.append(f"-_{mode}*{modal_lower}")
                complete_labels_list.append(f"+_{mode}*{modal_raise}")
        complete_label = " ".join(complete_labels_list)
        return complete_label
