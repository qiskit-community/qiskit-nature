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
import re
from typing import List, Tuple, Union

from qiskit.opflow import PauliSumOp


def read_expected_file(path: str):
    """Reads and parses resource file."""
    pauli_sum_op = 0
    with open(path, "r") as file:
        for line in file:
            pattern = re.compile(
                r"""(?P<sign>[+,-]?)\s?(?P<coefficient>\d*.\d*)\s\*\s(?P<operator>[I, Z]*)""",
                re.VERBOSE)
            match = pattern.match(line)
            sign = match.group("sign")
            coefficient = float(match.group("coefficient"))
            operator = match.group("operator")
            if sign == "-":
                coefficient = -coefficient
            pauli_sum_op += PauliSumOp.from_list([(operator, coefficient)])
    return pauli_sum_op
