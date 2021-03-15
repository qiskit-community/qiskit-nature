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
This module converts labels from the `VibrationalSpinOp` to 'SpinOp` notation.
"""
import re
from typing import List, Tuple, Union


def convert_to_spin_op_labels(vibrational_labels: List[Tuple[str, complex]], num_modes: int,
                              num_modals: Union[int, List[int]]) -> List[Tuple[str, complex]]:
    """Converts `VibrationalSpinOp` labels to `SpinOp` labels.

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

    if not _is_labels_valid(vibrational_labels, num_modes, num_modals):
        raise ValueError(
            "Provided labels are not valid - indexing out of range or non-matching raising "
            "and lowering operators per mode in a term")

    partial_sum_modals = _calc_partial_sum_modals(num_modals)

    spin_op_labels = []
    for labels, coeff in vibrational_labels:
        coeff_new_labels = _build_coeff_spin_op_labels(labels, partial_sum_modals)
        spin_op_labels.append((coeff_new_labels, coeff))
    return spin_op_labels


def _calc_partial_sum_modals(num_modals: Union[int, List[int]]) -> List[int]:
    summed = 0
    partial_sum_modals = [0]
    if isinstance(num_modals, list):
        for mode_len in num_modals:
            summed += mode_len
            partial_sum_modals.append(summed)
        return partial_sum_modals
    else:
        raise ValueError(f"num_modals of incorrect type {type(num_modals)}.")


def _build_coeff_spin_op_labels(labels: str, partial_sum_modals: List[int]):
    coeff_labels_split = labels.split(" ")
    coeff_new_labels = []
    for label in coeff_labels_split:
        new_label = _build_spin_op_label(label, partial_sum_modals)
        coeff_new_labels.append(new_label)
    coeff_new_labels = " ".join(coeff_new_labels)
    return coeff_new_labels


def _build_spin_op_label(label: str, partial_sum_modals: List[int]):
    op, mode_index, modal_index = re.split('[*_]', label)
    index = _get_ind_from_mode_modal(partial_sum_modals, int(mode_index), int(modal_index))
    new_label = "".join([op, "_", str(index)])
    return new_label


def _get_ind_from_mode_modal(partial_sum_modals: List[int], mode_index: int, modal_index: int):
    return partial_sum_modals[mode_index] + modal_index


def _is_labels_valid(vibrational_labels: List[Tuple[str, complex]], num_modes: int,
                     num_modals: Union[int, List[int]]):
    for labels, _ in vibrational_labels:
        coeff_labels_split = labels.split(" ")
        check_list = [0] * num_modes
        for label in coeff_labels_split:
            op, mode_index, modal_index = re.split('[*_]', label)
            if int(mode_index) >= num_modes or int(modal_index) >= num_modals[int(mode_index)]:
                return False
            increment = 1 if op == "+" else -1
            check_list[int(mode_index)] += increment
        if not all(v == 0 for v in check_list):
            return False
    return True
