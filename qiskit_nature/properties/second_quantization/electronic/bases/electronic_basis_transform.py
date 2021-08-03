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

"""The ElectronicBasisTransform provides a container of bases transformation data."""

from typing import Optional

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
        self.coeff_alpha = coeff_alpha
        self.coeff_beta = coeff_alpha if coeff_beta is None else coeff_beta

    # TODO: implement __str__
