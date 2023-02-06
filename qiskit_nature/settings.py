# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Nature Settings."""

from __future__ import annotations

import warnings


class ListAuxOpsDeprecationWarning(DeprecationWarning):
    """Deprecation Category for List-based aux. operators."""

    pass


class QiskitNatureSettings:
    """Global settings for Qiskit Nature."""

    def __init__(self) -> None:
        self._dict_aux_operators = True
        self._optimize_einsum = True
        self._deprecation_shown: set[str] = set()
        self._tensor_wrapping = False

    @property
    def dict_aux_operators(self) -> bool:
        """Return whether `aux_operators` are dictionary- or list-based."""
        if not self._dict_aux_operators and "ListAuxOps" not in self._deprecation_shown:
            warnings.filterwarnings("default", category=ListAuxOpsDeprecationWarning)
            warnings.warn(
                ListAuxOpsDeprecationWarning(
                    "List-based `aux_operators` are deprecated as of version 0.3.0 and support for "
                    "them will be removed no sooner than 3 months after the release. Instead, use "
                    "dict-based `aux_operators`. You can switch to the dict-based interface "
                    "immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`."
                ),
                stacklevel=3,
            )
            warnings.filterwarnings("ignore", category=ListAuxOpsDeprecationWarning)
            self._deprecation_shown.add("ListAuxOps")

        return self._dict_aux_operators

    @dict_aux_operators.setter
    def dict_aux_operators(self, dict_aux_operators: bool) -> None:
        """Set whether `aux_operators` are dictionary- or list-based."""
        if not dict_aux_operators and "ListAuxOps" not in self._deprecation_shown:
            warnings.filterwarnings("default", category=ListAuxOpsDeprecationWarning)
            warnings.warn(
                ListAuxOpsDeprecationWarning(
                    "List-based `aux_operators` are deprecated as of version 0.3.0 and support for "
                    "them will be removed no sooner than 3 months after the release. Instead, use "
                    "dict-based `aux_operators`. You can switch to the dict-based interface "
                    "immediately, by setting `qiskit_nature.settings.dict_aux_operators` to `True`."
                ),
                stacklevel=3,
            )
            warnings.filterwarnings("ignore", category=ListAuxOpsDeprecationWarning)
            self._deprecation_shown.add("ListAuxOps")

        self._dict_aux_operators = dict_aux_operators

    @property
    def optimize_einsum(self) -> bool:
        """Returns the setting used for `numpy.einsum(optimize=...)`.

        This is only used for calls with 3 or more operands. For more details refer to:
        https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        return self._optimize_einsum

    @optimize_einsum.setter
    def optimize_einsum(self, optimize_einsum: bool) -> None:
        """Sets the setting used for `numpy.einsum(optimize=...)`.

        This is only used for calls with 3 or more operands. For more details refer to:
        https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        self._optimize_einsum = optimize_einsum

    @property
    def tensor_wrapping(self) -> bool:
        """Returns whether tensors in Nature should be wrapped.

        More specifically, if this setting is enabled, tensors stored in a
        :class:`~qiskit_nature.second_q.operators.PolynomialTensor` will be wrapped into instances
        of :class:`~qiskit_nature.second_q.operators.Tensor`.
        """
        if not self._tensor_wrapping and "Tensor" not in self._deprecation_shown:
            warnings.filterwarnings("default", category=DeprecationWarning)
            warnings.warn(
                DeprecationWarning(
                    "As of version 0.6.0 the use of unwrapped arrays in the `PolynomialTensor` is "
                    "deprecated. No sooner that 3 months after this release, arrays will "
                    "automatically be wrapped into `Tensor` classes if not already done so by the "
                    "user. You can switch to the wrapping immediately, by setting "
                    "`qiskit_nature.settings.tensor_wrapping` to `True`."
                ),
                stacklevel=3,
            )
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self._deprecation_shown.add("Tensor")

        return self._tensor_wrapping

    @tensor_wrapping.setter
    def tensor_wrapping(self, tensor_wrapping: bool) -> None:
        """Returns whether tensors in Nature should be wrapped.

        More specifically, if this setting is enabled, tensors stored in a
        :class:`~qiskit_nature.second_q.operators.PolynomialTensor` will be wrapped into instances
        of :class:`~qiskit_nature.second_q.operators.Tensor`.
        """
        if not tensor_wrapping and "Tensor" not in self._deprecation_shown:
            warnings.filterwarnings("default", category=DeprecationWarning)
            warnings.warn(
                DeprecationWarning(
                    "As of version 0.6.0 the use of unwrapped arrays in the `PolynomialTensor` is "
                    "deprecated. No sooner that 3 months after this release, arrays will "
                    "automatically be wrapped into `Tensor` classes if not already done so by the "
                    "user. You can switch to the wrapping immediately, by setting "
                    "`qiskit_nature.settings.tensor_wrapping` to `True`."
                ),
                stacklevel=3,
            )
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self._deprecation_shown.add("Tensor")

        self._tensor_wrapping = tensor_wrapping


settings = QiskitNatureSettings()
