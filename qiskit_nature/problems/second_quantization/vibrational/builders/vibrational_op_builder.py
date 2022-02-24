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
""" Vibrational operator builder. """
from typing import List, Tuple

import logging

from qiskit_nature.deprecation import DeprecatedType, deprecate_function

from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.problems.second_quantization.vibrational.builders.vibrational_label_builder import (
    _create_labels,
)

logger = logging.getLogger(__name__)


@deprecate_function(
    "0.2.0",
    DeprecatedType.CLASS,
    "VibrationalEnergy",
    "from qiskit_nature.properties.second_quantization.vibrational in combination with the new "
    "HarmonicBasis from qiskit_nature.properties.second_quantization.vibrational.bases",
)
def build_vibrational_op_from_ints(
    h_mat: List[List[Tuple[List[List[int]], complex]]],
    num_modes: int,
    num_modals: List[int],
) -> VibrationalOp:
    """**DEPRECATED!**
    Builds a :class:`VibrationalOp` based on an integral list as produced by
    :meth:`HarmonicBasis.convert()`.

    Args:
        h_mat: integral list.
        num_modes: the number of modes.
        num_modals: the number of modals.

    Returns:
        The constructed VibrationalOp.
    """
    all_labels = _create_labels(h_mat)

    return VibrationalOp(all_labels, num_modes, num_modals)
