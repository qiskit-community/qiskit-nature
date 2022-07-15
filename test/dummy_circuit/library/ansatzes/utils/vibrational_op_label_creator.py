# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Some utility methods which were removed but are still required for some unit-tests."""

from typing import List, Tuple


def _create_labels(
    boson_hamilt_harm_basis: List[List[Tuple[List[List[int]], complex]]]
) -> List[Tuple[str, complex]]:
    all_labels = []
    for num_body_data in boson_hamilt_harm_basis:
        num_body_labels = _create_num_body_labels(num_body_data)
        all_labels.extend(num_body_labels)
    return all_labels


def _create_num_body_labels(
    num_body_data: List[Tuple[List[List[int]], complex]]
) -> List[Tuple[str, complex]]:
    num_body_labels = []
    for indices, coeff in num_body_data:
        indices.sort()
        coeff_label = _create_label_for_coeff(indices)
        num_body_labels.append((coeff_label, coeff))
    return num_body_labels


def _create_label_for_coeff(indices: List[List[int]]) -> str:
    complete_labels_list = []
    for mode, modal_raise, modal_lower in indices:
        if modal_raise <= modal_lower:
            complete_labels_list.append(f"+_{mode}*{modal_raise}")
            complete_labels_list.append(f"-_{mode}*{modal_lower}")
        else:
            complete_labels_list.append(f"-_{mode}*{modal_lower}")
            complete_labels_list.append(f"+_{mode}*{modal_raise}")
    complete_label = " ".join(complete_labels_list)
    return complete_label
