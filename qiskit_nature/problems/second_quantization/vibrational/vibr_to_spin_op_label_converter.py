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


def calc_partial_sum_modals(num_modes, num_modals):
    # TODO docs
    summed = 0
    partial_sum_modals = [0]
    if type(num_modals) == list:
        for mode_len in num_modals:
            summed += mode_len
            partial_sum_modals.append(summed)
        return partial_sum_modals
    elif type(num_modals) == int:
        for _ in range(num_modes):
            summed += num_modals
            partial_sum_modals.append(summed)
        return partial_sum_modals
    else:
        raise ValueError(f"num_modals of incorrect type {type(num_modals)}.")


def convert_to_spin_op_labels(data, partial_sum_modals):
    # TODO docs
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
