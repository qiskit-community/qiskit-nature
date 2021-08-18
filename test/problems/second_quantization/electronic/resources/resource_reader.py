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

"""Utility resource methods."""

from typing import List, Tuple, Union


def read_expected_file(path: str) -> List[Tuple[Union[str, float], ...]]:
    """Reads and parses resource file."""
    expected_fermionic_op = []
    with open(path, "r") as file:
        for line in file:
            coeff, *labels = line.split()
            expected_fermionic_op.append((" ".join(labels), float(coeff)))
    return expected_fermionic_op
