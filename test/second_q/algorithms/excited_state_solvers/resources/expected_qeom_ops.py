# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Reference SparsePauliOp for testing."""

from typing import Any, List, Dict
from qiskit.quantum_info import SparsePauliOp

expected_hopping_operators_electronic: Dict[str, SparsePauliOp] = {
    "E_0": 1 / 4 * SparsePauliOp.from_list([("IIXY", -1j), ("IIYY", 1), ("IIXX", 1), ("IIYX", 1j)]),
    "Edag_0": 1
    / 4
    * SparsePauliOp.from_list([("IIXY", 1j), ("IIXX", 1), ("IIYY", 1), ("IIYX", -1j)]),
    "E_1": 1 / 4 * SparsePauliOp.from_list([("XYII", -1j), ("YYII", 1), ("XXII", 1), ("YXII", 1j)]),
    "Edag_1": 1
    / 4
    * SparsePauliOp.from_list([("XYII", 1j), ("YYII", 1), ("XXII", 1), ("YXII", -1j)]),
    "E_2": 1
    / 16
    * SparsePauliOp.from_list(
        [
            ("XYXY", 1),
            ("YYXY", 1j),
            ("XYYY", 1j),
            ("YYYY", -1),
            ("XXXY", 1j),
            ("YXXY", -1),
            ("XXYY", -1),
            ("YXYY", -1j),
            ("XYXX", 1j),
            ("YYXX", -1),
            ("XYYX", -1),
            ("YYYX", -1j),
            ("XXXX", -1),
            ("YXXX", -1j),
            ("XXYX", -1j),
            ("YXYX", 1),
        ]
    ),
    "Edag_2": 1
    / 16
    * SparsePauliOp.from_list(
        [
            ("XYXY", 1),
            ("XXXY", -1j),
            ("XYXX", -1j),
            ("XXXX", -1),
            ("YYXY", -1j),
            ("YXXY", -1),
            ("YYXX", -1),
            ("YXXX", 1j),
            ("XYYY", -1j),
            ("XXYY", -1),
            ("XYYX", -1),
            ("XXYX", 1j),
            ("YYYY", -1),
            ("YXYY", 1j),
            ("YYYX", 1j),
            ("YXYX", 1),
        ]
    ),
}

expected_commutativies_electronic: Dict[str, List[bool]] = {
    "E_0": [],
    "Edag_0": [],
    "E_1": [],
    "Edag_1": [],
    "E_2": [],
    "Edag_2": [],
}

expected_indices_electronic: Dict[str, Any] = {
    "E_0": ((0,), (1,)),
    "Edag_0": ((1,), (0,)),
    "E_1": ((2,), (3,)),
    "Edag_1": ((3,), (2,)),
    "E_2": ((0, 2), (1, 3)),
    "Edag_2": ((1, 3), (0, 2)),
}

expected_hopping_operators_vibrational: Dict[str, SparsePauliOp] = {
    "E_0": SparsePauliOp.from_list(
        [("IIXX", 0.25), ("IIYX", 0.25j), ("IIXY", -0.25j), ("IIYY", 0.25)]
    ),
    "Edag_0": SparsePauliOp.from_list(
        [("IIXX", 0.25), ("IIYX", -0.25j), ("IIXY", 0.25j), ("IIYY", 0.25)]
    ),
    "E_1": SparsePauliOp.from_list(
        [("XXII", 0.25), ("YXII", 0.25j), ("XYII", -0.25j), ("YYII", 0.25)]
    ),
    "Edag_1": SparsePauliOp.from_list(
        [("XXII", 0.25), ("YXII", -0.25j), ("XYII", 0.25j), ("YYII", 0.25)]
    ),
    "E_2": SparsePauliOp.from_list(
        [
            ("XXXX", 0.0625),
            ("YXXX", 0.0625j),
            ("XYXX", -0.0625j),
            ("YYXX", 0.0625),
            ("XXYX", 0.0625j),
            ("YXYX", -0.0625),
            ("XYYX", 0.0625),
            ("YYYX", 0.0625j),
            ("XXXY", -0.0625j),
            ("YXXY", 0.0625),
            ("XYXY", -0.0625),
            ("YYXY", -0.0625j),
            ("XXYY", 0.0625),
            ("YXYY", 0.0625j),
            ("XYYY", -0.0625j),
            ("YYYY", 0.0625),
        ]
    ),
    "Edag_2": SparsePauliOp.from_list(
        [
            ("XXXX", 0.0625),
            ("YXXX", -0.0625j),
            ("XYXX", 0.0625j),
            ("YYXX", 0.0625),
            ("XXYX", -0.0625j),
            ("YXYX", -0.0625),
            ("XYYX", 0.0625),
            ("YYYX", -0.0625j),
            ("XXXY", 0.0625j),
            ("YXXY", 0.0625),
            ("XYXY", -0.0625),
            ("YYXY", 0.0625j),
            ("XXYY", 0.0625),
            ("YXYY", -0.0625j),
            ("XYYY", 0.0625j),
            ("YYYY", 0.0625),
        ]
    ),
}
expected_commutativies_vibrational: Dict[str, List[bool]] = {}
expected_indices_vibrational: Dict[str, Any] = {
    "E_0": ((0,), (1,)),
    "Edag_0": ((1,), (0,)),
    "E_1": ((2,), (3,)),
    "Edag_1": ((3,), (2,)),
    "E_2": ((0, 2), (1, 3)),
    "Edag_2": ((1, 3), (0, 2)),
}
