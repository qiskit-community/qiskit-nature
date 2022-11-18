# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimized einsum utilities."""

from __future__ import annotations

from typing import Callable
import numpy as np
import qiskit_nature.optionals as _optionals


def get_einsum() -> tuple[Callable, bool]:
    """Returns tuple einsum function and flag indicating if dense should be applied
    to parameters.

    Returns:
        Tuple with function and if dense should be applied to parameters.
    """
    if _optionals.HAS_OPT_EINSUM:
        # pylint: disable=import-error
        from opt_einsum import contract

        return contract, False

    return np.einsum, True
