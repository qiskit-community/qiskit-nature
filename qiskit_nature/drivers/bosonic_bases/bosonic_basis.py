# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Bosonic Basis """

from typing import List, Tuple

import warnings


class BosonicBasis:
    """**DEPRECATED** Basis to express a second quantization Bosonic Hamiltonian."""

    def __init__(self):
        warnings.warn(
            "This BosonicBasis is deprecated as of 0.2.0, "
            "and will be removed no earlier than 3 months after the release. "
            "You should use the qiskit_nature.drivers.second_quantization.bosonic_bases "
            "BosonicBasis as a direct replacement instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def convert(self, threshold: float = 1e-6) -> List[List[Tuple[List[List[int]], complex]]]:
        """
        This prepares an array object representing a bosonic hamiltonian expressed
        in the harmonic basis. This object can directly be given to the BosonicOperator
        class to be mapped to a qubit hamiltonian.

        Args:
            threshold: the matrix elements of value below this threshold are discarded

        Returns:
            List of modes for input to creation of a bosonic hamiltonian in the harmonic basis

        Raises:
            ValueError: If problem with order value from computed modes
        """

        raise NotImplementedError
