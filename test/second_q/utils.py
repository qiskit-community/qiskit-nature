# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Some testing utilities."""

from qiskit_nature.second_q.operators.symmetric_two_body import (
    S1Integrals,
    S4Integrals,
    S8Integrals,
    fold_s1_to_s4,
    fold_s1_to_s8,
)
from qiskit_nature.second_q.operators.tensor_ordering import _chem_to_phys


def get_expected_two_body_ints(actual_ints, expected_ints):
    """Returns the ``expected_ints`` with the type of ``actual_ints``."""
    if isinstance(actual_ints, S1Integrals):
        return S1Integrals(expected_ints).to_dense()
    elif isinstance(actual_ints, S4Integrals):
        return fold_s1_to_s4(expected_ints, validate=False)
    elif isinstance(actual_ints, S8Integrals):
        return fold_s1_to_s8(expected_ints, validate=False)
    return _chem_to_phys(expected_ints)
