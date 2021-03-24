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
"""
This module converts labels from the `VibrationalOp` to 'SpinOp` notation.
"""
import itertools
import operator
import re
from typing import List, Tuple, Union

from .vibrational_labels_validator import _validate_vibrational_labels


def _convert_to_dense_labels(vibrational_labels: List[Tuple[str, complex]], num_modes: int,
                             num_modals: Union[int, List[int]]) -> List[Tuple[str, complex]]:
    """Converts `VibrationalOp` labels to `SpinOp` labels.

    Args:
        vibrational_labels: list of labels and corresponding coefficients that describe a
        vibrational problem.
        num_modes: number of modes.
        num_modals: number of modals in each mode.
    Returns:
        A list of labels and corresponding coefficients that describe a
        vibrational problem in terms of labels accepted by `SpinOp`.
    Raises:
        ValueError: if invalid labels provided or the length of list of modal sizes do not agree
        with the number of modes provided
    """
    if isinstance(num_modals, int):
        num_modals = [num_modals] * num_modes

    if len(num_modals) != num_modes:
        raise ValueError("num_modes does not agree with the size of num_modals")

    _validate_vibrational_labels(vibrational_labels, num_modes, num_modals)

    partial_sum_modals = [0] + list(itertools.accumulate(num_modals, operator.add))

    dense_labels = []
    for labels, coeff in vibrational_labels:
        coeff_new_labels = _build_coeff_dense_labels(labels, partial_sum_modals)
        dense_labels.append((coeff_new_labels, coeff))
    return dense_labels


def _build_coeff_dense_labels(labels: str, partial_sum_modals: List[int]) -> str:
    coeff_labels_split = labels.split()
    coeff_new_labels = []
    for label in coeff_labels_split:
        op, index = _build_dense_label(label, partial_sum_modals)
        new_label = ['I'] * partial_sum_modals[-1]
        new_label[index] = op
        coeff_new_labels.append(''.join(new_label))
    return coeff_new_labels


def _build_dense_label(label: str, partial_sum_modals: List[int]) -> str:
    op, mode_index, modal_index = re.split('[*_]', label)
    index = _get_ind_from_mode_modal(partial_sum_modals, int(mode_index), int(modal_index))
    return (op, index)


def _get_ind_from_mode_modal(partial_sum_modals: List[int], mode_index: int,
                             modal_index: int) -> int:
    return partial_sum_modals[mode_index] + modal_index
