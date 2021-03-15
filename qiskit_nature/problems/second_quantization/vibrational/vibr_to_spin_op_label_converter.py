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


def _calc_partial_sum_modals(num_modals):
    # TODO docs

    summed = 0
    partial_sum_modals = [0]
    if type(num_modals) == list:
        for mode_len in num_modals:
            summed += mode_len
            partial_sum_modals.append(summed)
        return partial_sum_modals
    else:
        raise ValueError(f"num_modals of incorrect type {type(num_modals)}.")


def convert_to_spin_op_labels(data, num_modes, num_modals):
    # TODO docs
    if type(num_modals) == int:
        num_modals = [num_modals]*num_modes

    if len(num_modals) != num_modes:
        raise ValueError("num_modes does not agree with the size of num_modals")
    if not _is_labels_valid(data, num_modes, num_modals):
        raise ValueError(
            "Provided labels are not valid - indexing out of range or non-matching raising "
            "and lowering operators per mode in a term")

    partial_sum_modals = _calc_partial_sum_modals(num_modals)

    spin_op_labels = []
    for labels, coeff in data:
        coeff_new_labels = _build_coeff_spin_op_labels(labels, partial_sum_modals)
        spin_op_labels.append((coeff_new_labels, coeff))
    return spin_op_labels


def _build_coeff_spin_op_labels(labels, partial_sum_modals):
    coeff_labels_split = labels.split(" ")
    coeff_new_labels = []
    for label in coeff_labels_split:
        new_label = _build_spin_op_label(label, partial_sum_modals)
        coeff_new_labels.append(new_label)
    coeff_new_labels = " ".join(coeff_new_labels)
    return coeff_new_labels


def _build_spin_op_label(label, partial_sum_modals):
    op, mode_index, modal_index = re.split('[*_]', label)
    index = _get_ind_from_mode_modal(partial_sum_modals, int(mode_index), int(modal_index))
    new_label = "".join([op, "_", str(index)])
    return new_label


def _get_ind_from_mode_modal(partial_sum_modals, mode_index, modal_index):
    return partial_sum_modals[mode_index] + modal_index


def _is_labels_valid(vibrational_data, num_modes, num_modals):
    for labels, coeff in vibrational_data:
        coeff_labels_split = labels.split(" ")
        check_list = [0] * num_modes
        for label in coeff_labels_split:
            op, mode_index, modal_index = re.split('[*_]', label)
            if int(mode_index) >= num_modes or int(modal_index) >= num_modals[
                int(mode_index)]:
                return False
            increment = 1 if op == "+" else -1
            check_list[int(mode_index)] += increment
        if not all(v == 0 for v in check_list):

            return False
    return True
