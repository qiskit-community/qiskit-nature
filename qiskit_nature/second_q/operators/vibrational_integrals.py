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

"""A container class for vibrational operator coefficients."""

from __future__ import annotations

from typing import Mapping

import logging

import numpy as np

import qiskit_nature.optionals as _optionals

from .polynomial_tensor import PolynomialTensor
from .tensor import Tensor

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import as_coo
else:

    def as_coo(*args):
        """Empty as_coo function
        Replacement if sparse.as_coo is not present.
        """
        del args


LOGGER = logging.getLogger(__name__)


class VibrationalIntegrals(PolynomialTensor):
    """A container class for vibrational operator coefficients (a.k.a. vibrational integrals).

    This class simply derives the :class:`qiskit_nature.second_q.operators.PolynomialTensor`
    implementation and provides an efficient factory method from a raw set of integrals.
    Being a subclass of the ``PolynomialTensor``, the same arithmetic operations are supported as by
    that class.
    """

    @classmethod
    def from_raw_integrals(
        cls, integrals: Mapping[tuple[int, ...], complex]
    ) -> VibrationalIntegrals:
        """Constructs a ``VibrationalIntegrals`` instance from the provided coefficients.

        The provided coefficients must already be mapped to a second-quantization basis. See the
        documentation of :class:`qiskit_nature.second_q.problems.VibrationalBasis` for more details.

        Args:
            integrals: a mapping of matrix index tuples to coefficients. This is effectively a
                sparse representation of the coefficients. Each key in the mapping should be a tuple
                of integers of the form ``(mode, modal_1, modal_2, ...)``. That means that each key
                is expected to have its length be a multiple of three. See also the documentation of
                :meth:`qiskit_nature.second_q.problems.VibrationalBasis.map` for more details.

        Returns:
            The constructed instance.
        """
        if _optionals.HAS_SPARSE:
            max_n_body = max(len(key) for key in integrals) // 3
            ret = cls(
                {
                    ("_+-" * n_body): Tensor(
                        as_coo({k: v for k, v in integrals.items() if len(k) == 3 * n_body}),
                        label_template=" ".join(["{}_{{}}_{{}}"] * n_body * 2),
                    )
                    for n_body in range(1, max_n_body + 1)
                },
                validate=False,
            )
        else:
            LOGGER.warning(
                "The optional dependency 'sparse' is not installed. Falling back to using 'numpy' "
                "arrays instead. Consider installing the 'sparse' package to reduce memory "
                "requirements."
            )
            data = {}
            max_n_body = 0
            max_mode = 0
            max_modal = 0
            for key in integrals:
                max_n_body = max(max_n_body, len(key) // 3)
                max_mode = max(max_mode, *key[::3])
                max_modal = max(max_modal, *key[1::3], *key[2::3])

            max_mode += 1
            max_modal += 1
            for n_body in range(1, max_n_body + 1):
                data_key = "_+-" * n_body
                numpy_arr = np.zeros((max_mode, max_modal, max_modal) * n_body, dtype=complex)
                for k, v in integrals.items():
                    if len(k) == 3 * n_body:
                        numpy_arr[k] = v
                data[data_key] = Tensor(
                    numpy_arr, label_template=" ".join(["{}_{{}}_{{}}"] * n_body * 2)
                )
            ret = cls(data, validate=False)
        return ret

    @property
    def register_length(self) -> int | None:
        # for now, we simply return None here.
        return None
