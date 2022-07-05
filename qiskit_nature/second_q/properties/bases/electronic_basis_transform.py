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

"""The ElectronicBasisTransform provides a container of bases transformation data."""

from __future__ import annotations

from typing import Optional

import h5py

import numpy as np

from .electronic_basis import ElectronicBasis
from ..property import Property


class ElectronicBasisTransform(Property):
    """This class contains the coefficients required to map from one basis into another."""

    def __init__(
        self,
        initial_basis: ElectronicBasis,
        final_basis: ElectronicBasis,
        coeff_alpha: np.ndarray,
        coeff_beta: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            initial_basis: the initial basis from which to map out of.
            final_basis: the final basis which to map in to.
            coeff_alpha: the matrix (``# orbitals in initial basis x # orbitals in final basis``)
                for mapping the alpha-spin orbitals.
            coeff_beta: an optional matrix to use for the beta-spin orbitals. This must match the
                dimension of ``coeff_alpha``. If it is left as ``None``, ``coeff_alpha`` will be
                used for the beta-spin orbitals too.
        """
        super().__init__(self.__class__.__name__)
        self.initial_basis = initial_basis
        self.final_basis = final_basis
        self._coeff_alpha = coeff_alpha
        self._coeff_beta = coeff_beta

    @property
    def coeff_alpha(self) -> np.ndarray:
        """Returns the alpha-spin coefficient matrix."""
        return self._coeff_alpha

    @property
    def coeff_beta(self) -> np.ndarray:
        """Returns the beta-spin coefficient matrix."""
        return self._coeff_beta if self._coeff_beta is not None else self._coeff_alpha

    def is_alpha_equal_beta(self) -> bool:
        """Returns whether the alpha- and beta-spin coefficient matrices are close."""
        return np.allclose(self.coeff_alpha, self.coeff_beta)

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\tInitial basis: {self.initial_basis.value}"]
        string += [f"\tFinal basis: {self.final_basis.value}"]
        string += ["\tAlpha coefficients:"]
        string += self._render_coefficients(self.coeff_alpha)
        if self._coeff_beta is not None:
            string += ["\tBeta coefficients:"]
            string += self._render_coefficients(self.coeff_beta)
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["initial_basis"] = self.initial_basis.name
        group.attrs["final_basis"] = self.final_basis.name

        group.create_dataset("Alpha coefficients", data=self.coeff_alpha)
        if not self.is_alpha_equal_beta():
            group.create_dataset("Beta coefficients", data=self.coeff_beta)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> ElectronicBasisTransform:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        coeff_alpha = h5py_group["Alpha coefficients"][...]
        coeff_beta: Optional[np.ndarray] = None
        if "Beta coefficients" in h5py_group.keys():
            coeff_beta = h5py_group["Beta coefficients"][...]
        return ElectronicBasisTransform(
            getattr(ElectronicBasis, h5py_group.attrs["initial_basis"]),
            getattr(ElectronicBasis, h5py_group.attrs["final_basis"]),
            coeff_alpha,
            coeff_beta,
        )

    @staticmethod
    def _render_coefficients(coeffs) -> list[str]:
        nonzero = coeffs.nonzero()
        return [f"\t{indices} = {value}" for value, *indices in zip(coeffs[nonzero], *nonzero)]
