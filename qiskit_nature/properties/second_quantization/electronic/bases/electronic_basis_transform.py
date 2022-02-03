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

from typing import List, Optional

import numpy as np

from ....property import PseudoProperty
from .electronic_basis import ElectronicBasis


class ElectronicBasisTransform(PseudoProperty):
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

    @staticmethod
    def _render_coefficients(coeffs) -> List[str]:
        nonzero = coeffs.nonzero()
        return [f"\t{indices} = {value}" for value, *indices in zip(coeffs[nonzero], *nonzero)]
