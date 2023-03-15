# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Watson Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

import qiskit_nature.optionals as _optionals

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray, as_coo
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass

    def as_coo(*args):
        """Empty as_coo function
        Replacement if sparse.as_coo is not present.
        """
        del args


@dataclass
class WatsonHamiltonian:
    r"""A dataclass for the force and kinetic coefficients describing a Watson Hamiltonian.

    .. math::
        \mathcal{H}_{vib}(Q_1, \ldots, Q_L) =
            - \frac{1}{2} \sum_{l=1}^L \frac{\partial^2}{\partial Q_l^2}
            + V(Q_1, \ldots, Q_L)
    """

    quadratic_force_constants: SparseArray | np.ndarray
    """The quadratic-order force constants."""

    cubic_force_constants: SparseArray | np.ndarray
    """The cubic-order force constants."""

    quartic_force_constants: SparseArray | np.ndarray
    """The quartic-order force constants."""

    kinetic_coefficients: SparseArray | np.ndarray
    """The kinetic coefficients."""

    @staticmethod
    def _iter_array(
        array: SparseArray | np.ndarray,
        *,
        kinetic: bool = False,
    ) -> Generator[tuple[complex, tuple[int, ...]], None, None]:
        if isinstance(array, np.ndarray):
            for index in np.ndindex(*array.shape):
                value = array[index]
                if value:
                    yield value, tuple((-1) ** kinetic * (i + 1) for i in index)
        elif isinstance(array, SparseArray):
            coo = as_coo(array)
            for value, *index in zip(coo.data, *coo.coords):
                yield value, tuple((-1) ** kinetic * (i + 1) for i in index)

    def __iter__(self) -> Generator[tuple[complex, tuple[int, ...]], None, None]:
        for value, index in self._iter_array(self.quadratic_force_constants):
            yield value, index
        for value, index in self._iter_array(self.cubic_force_constants):
            yield value, index
        for value, index in self._iter_array(self.quartic_force_constants):
            yield value, index
        for value, index in self._iter_array(self.kinetic_coefficients, kinetic=True):
            yield value, index
