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
import re
from typing import List, Tuple, Union


def validate_vibrational_labels(vibrational_labels: List[Tuple[str, complex]], num_modes: int,
                                num_modals: Union[int, List[int]]):
    if isinstance(num_modals, int):
        num_modals = [num_modals] * num_modes
    _validate_data_type(vibrational_labels)
    _validate_regex(vibrational_labels)
    _validate_indices(vibrational_labels, num_modes, num_modals)


def _validate_data_type(vibrational_labels: List[Tuple[str, complex]]):
    if not isinstance(vibrational_labels, list):
        raise ValueError("Invalid data type.")


def _validate_regex(vibrational_labels: List[Tuple[str, complex]]):
    valid_vibr_label_pattern = re.compile(r"^([\+\-]_\d+\*\d+\s)*[\+\-]_\d+\*\d+(?!\s)$|^[\+\-]+$")
    invalid_labels = [label for label, _ in vibrational_labels if
                      not valid_vibr_label_pattern.match(label)]
    if invalid_labels:
        raise ValueError(f"Invalid labels: {invalid_labels}")


def _validate_indices(vibrational_labels: List[Tuple[str, complex]], num_modes: int,
                      num_modals: Union[int, List[int]]):
    for labels, _ in vibrational_labels:
        coeff_labels_split = labels.split(" ")
        check_list = [0] * num_modes
        last_op, last_mode_index, last_modal_index = "+", num_modes, max(num_modals)
        for label in coeff_labels_split:
            op, mode_index, modal_index = re.split('[*_]', label)
            mode_index, modal_index = int(mode_index), int(modal_index)
            if _is_index_out_of_range(mode_index, num_modes, modal_index, num_modals):
                raise ValueError(f"Indices out of the declared range for label {label}.")
            if _is_label_duplicated(mode_index, last_mode_index, modal_index, last_modal_index, op,
                                    last_op):
                raise ValueError(f"Operators in a label duplicated for label {label}.")
            if _is_order_incorrect(mode_index, last_mode_index, modal_index, last_modal_index, op,
                                   last_op):
                raise ValueError(
                    f"Incorrect order of operators for label {label} and previous label "
                    f"{str(last_op) + str('_') + str(last_mode_index) + str('*') + str(last_modal_index)}.")

            last_op, last_mode_index, last_modal_index = op, mode_index, modal_index

            increment = 1 if op == "+" else -1
            check_list[int(mode_index)] += increment
        if not all(v == 0 for v in check_list):
            raise ValueError(
                f"Modes of raising and lowering operators do not agree for labels {labels}.")


def _is_index_out_of_range(mode_index, num_modes, modal_index, num_modals):
    return int(mode_index) >= num_modes or int(modal_index) >= num_modals[int(mode_index)]


def _is_label_duplicated(mode_index, last_mode_index, modal_index, last_modal_index, op, last_op):
    return modal_index == last_modal_index and mode_index == last_mode_index and op == last_op


def _is_order_incorrect(mode_index, last_mode_index, modal_index, last_modal_index, op, last_op):
    return _is_mode_order_incorrect(mode_index, last_mode_index) or \
           _is_modal_order_incorrect(last_mode_index, mode_index, last_modal_index, modal_index) \
           or _is_operator_order_incorrect(mode_index, last_mode_index, modal_index,
                                           last_modal_index, op, last_op)


def _is_mode_order_incorrect(mode_index, last_mode_index):
    return int(mode_index) > last_mode_index


def _is_modal_order_incorrect(last_mode_index, mode_index, last_modal_index, modal_index):
    return int(mode_index) == last_mode_index and int(
        modal_index) > last_modal_index


def _is_operator_order_incorrect(mode_index, last_mode_index, modal_index, last_modal_index,
                                 op, last_op):
    return int(mode_index) == last_mode_index and int(
        modal_index) == last_modal_index and last_op == "-" and op == "+"
