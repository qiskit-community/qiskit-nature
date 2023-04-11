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


class PauliSumOpDeprecationWarning(DeprecationWarning):
    """Deprecation Category for PauliSumOp."""

    pass


class QiskitNatureSettings:
    """Global settings for Qiskit Nature."""

    def __init__(self) -> None:
        self._dict_aux_operators = True
        self._optimize_einsum = True
        self._deprecation_shown: set[str] = set()
        self._tensor_unwrapping = True
        self._use_pauli_sum_op: bool = True
        self._use_symmetry_reduced_integrals: bool = False

    @property
    def use_pauli_sum_op(self) -> bool:
        """Return whether ``PauliSumOp`` or ``SparsePauliOp`` should be returned on methods."""
        if self._use_pauli_sum_op and "PauliSumOp" not in self._deprecation_shown:
            warnings.filterwarnings("default", category=PauliSumOpDeprecationWarning)
            warnings.warn(
                PauliSumOpDeprecationWarning(
                    "PauliSumOp is deprecated as of version 0.6.0 and support for "
                    "them will be removed no sooner than 3 months after the release. Instead, use "
                    "SparsePauliOp. You can switch to SparsePauliOp "
                    "immediately, by setting `qiskit_nature.settings.use_pauli_sum_op` to `False`."
                ),
                stacklevel=3,
            )
            warnings.filterwarnings("ignore", category=PauliSumOpDeprecationWarning)
            self._deprecation_shown.add("PauliSumOp")

        return self._use_pauli_sum_op

    @use_pauli_sum_op.setter
    def use_pauli_sum_op(self, pauli_sum_op: bool) -> None:
        """Set whether ``PauliSumOp`` or ``SparsePauliOp`` should be returned on methods."""
        if pauli_sum_op and "PauliSumOp" not in self._deprecation_shown:
            warnings.filterwarnings("default", category=PauliSumOpDeprecationWarning)
            warnings.warn(
                PauliSumOpDeprecationWarning(
                    "PauliSumOp is deprecated as of version 0.6.0 and support for "
                    "them will be removed no sooner than 3 months after the release. Instead, use "
                    "SparsePauliOp. You can switch to SparsePauliOp "
                    "immediately, by setting `qiskit_nature.settings.use_pauli_sum_op` to `False`."
                ),
                stacklevel=3,
            )
            warnings.filterwarnings("ignore", category=PauliSumOpDeprecationWarning)
            self._deprecation_shown.add("PauliSumOp")
        self._use_pauli_sum_op = pauli_sum_op

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
    def use_symmetry_reduced_integrals(self) -> bool:
        """Whether or not to use symmetry-reduced integrals whenever possible.

        This setting affects whether the drivers and formats should attempt to use the utilities
        provided by the :mod:`~qiskit_nature.second_q.operators.symmetric_two_body` module.
        Setting this to ``True`` will very likely result in lower memory consumptions at runtime.
        """
        if (
            not self._use_symmetry_reduced_integrals
            and "SymmetricTwoBodyIntegrals" not in self._deprecation_shown
        ):
            warnings.warn(
                DeprecationWarning(
                    "As of version 0.6.0 the current default-value `False` of "
                    "`qiskit_nature.settings.use_symmetry_reduced_integrals` is deprecated. "
                    "No sooner than 3 months after this release, this default value will be "
                    "switched to `True`. You can change the value of this setting yourself already."
                ),
                stacklevel=3,
            )
            self._deprecation_shown.add("SymmetricTwoBodyIntegrals")

        return self._use_symmetry_reduced_integrals

    @use_symmetry_reduced_integrals.setter
    def use_symmetry_reduced_integrals(self, use_symmetry_reduced_integrals: bool) -> None:
        if (
            not use_symmetry_reduced_integrals
            and "SymmetricTwoBodyIntegrals" not in self._deprecation_shown
        ):
            warnings.warn(
                DeprecationWarning(
                    "As of version 0.6.0 the current default-value `False` of "
                    "`qiskit_nature.settings.use_symmetry_reduced_integrals` is deprecated. "
                    "No sooner than 3 months after this release, this default value will be "
                    "switched to `True`. You can change the value of this setting yourself already."
                ),
                stacklevel=3,
            )
            self._deprecation_shown.add("SymmetricTwoBodyIntegrals")

        self._use_symmetry_reduced_integrals = use_symmetry_reduced_integrals

    @property
    def tensor_unwrapping(self) -> bool:
        """Returns whether tensors inside the :class:`~.PolynomialTensor` should be unwrapped.

        More specifically, if this setting is disabled, the tensor objects stored in a
        :class:`~qiskit_nature.second_q.operators.PolynomialTensor` will be of type
        :class:`~qiskit_nature.second_q.operators.Tensor` when accessed via ``__getitem__``.
        Otherwise, they will appear as the nested array object which may be of type
        ``numpy.ndarray``, ``sparse.SparseArray`` or a plain ``Number``.
        """
        if self._tensor_unwrapping and "Tensor" not in self._deprecation_shown:
            warnings.warn(
                DeprecationWarning(
                    "As of version 0.6.0 the return of unwrapped tensors in the "
                    "`PolynomialTensor.__getitem__` method is deprecated. No sooner than 3 months "
                    "after this release, arrays will always be returned as `Tensor` objects. You "
                    "can switch to the new objects immediately, by setting "
                    "`qiskit_nature.settings.tensor_unwrapping` to `False`."
                ),
                stacklevel=3,
            )
            self._deprecation_shown.add("Tensor")

        return self._tensor_unwrapping

    @tensor_unwrapping.setter
    def tensor_unwrapping(self, tensor_unwrapping: bool) -> None:
        """Returns whether tensors inside the :class:`~.PolynomialTensor` should be unwrapped.

        More specifically, if this setting is disabled, the tensor objects stored in a
        :class:`~qiskit_nature.second_q.operators.PolynomialTensor` will be of type
        :class:`~qiskit_nature.second_q.operators.Tensor` when accessed via ``__getitem__``.
        Otherwise, they will appear as the nested array object which may be of type
        ``numpy.ndarray``, ``sparse.SparseArray`` or a plain ``Number``.
        """
        if tensor_unwrapping and "Tensor" not in self._deprecation_shown:
            warnings.warn(
                DeprecationWarning(
                    "As of version 0.6.0 the return of unwrapped tensors in the "
                    "`PolynomialTensor.__getitem__` method is deprecated. No sooner than 3 months "
                    "after this release, arrays will always be returned as `Tensor` objects. You "
                    "can switch to the new objects immediately, by setting "
                    "`qiskit_nature.settings.tensor_unwrapping` to `False`."
                ),
                stacklevel=3,
            )
            self._deprecation_shown.add("Tensor")

        self._tensor_unwrapping = tensor_unwrapping


settings = QiskitNatureSettings()
