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

"""Utility methods for the creation of common auxiliary operators."""

from typing import List

from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.problems.second_quantization.vibrational.integrals_calculators import (
    calc_occ_modals_per_mode_ints,
)
from .vibrational_op_builder import build_vibrational_op_from_ints


def _create_all_aux_operators(num_modals: List[int]) -> List[VibrationalOp]:
    """Generates the common auxiliary operators out of the given WatsonHamiltonian.

    Args:
        num_modals: the number of modals per mode.

    Returns:
        A list of VibrationalOps. For each mode the number of occupied modals will be evaluated.
    """
    aux_second_quantized_ops_list = []

    for mode in range(len(num_modals)):
        aux_second_quantized_ops_list.append(
            _create_occ_modals_per_mode(num_modals, mode)
        )

    return aux_second_quantized_ops_list


def _create_occ_modals_per_mode(
    num_modals: List[int], mode_index: int
) -> VibrationalOp:
    return build_vibrational_op_from_ints(
        calc_occ_modals_per_mode_ints(num_modals, mode_index),
        len(num_modals),
        num_modals,
    )
