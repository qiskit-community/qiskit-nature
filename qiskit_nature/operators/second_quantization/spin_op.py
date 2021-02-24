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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the `FermionicOperator` and `BosonicOperator`
as it relies an the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

import re
from functools import lru_cache, reduce
from itertools import product
from typing import cast, List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from .particle_op import ParticleOp


class SpinOp(ParticleOp):
    """Spin type operators. A class for products and powers of XYZ-ordered Spin operators.

    **Label**

    Allowed characters for primitives of labels are I, X, Y, Z, +, and -.

    .. list-table::
        :header-rows: 1

        * - Label
          - Mathematical Representation
          - Meaning
        * - `I`
          - :math:`I`
          - Identity operator
        * - `X`
          - :math:`S_x`
          - :math:`x`-component of the spin operator
        * - `Y`
          - :math:`S_y`
          - :math:`y`-component of the spin operator
        * - `Z`
          - :math:`S_z`
          - :math:`z`-component of the spin operator
        * - `+`
          - :math:`S_+`
          - Raising operator
        * - `-`
          - :math:`S_-`
          - Lowering operator

    There are two types of label modes for :class:`SpinOp`.

    1. Sparse Label (default)

    Sparse label mode is space-separated label.
    Each element is string consisting of :code:`[XYZI+-]_<index>^<power>`,
    where the :code:`<index>` is non-negative integer and the :code:`<power>` is positive integer.
    If :code:`<power>` is 1, it can be omitted.
    For example,

    .. code-block:: python

        "X_0"
        "Y_0^2"
        "Y_1^2 Z_1^3 X_0^1 Y_0^2 Z_0^2"

    are possible labels.
    `X`, `Y`, `Z` must be in this order for each index, so :code:`"Z_0 X_0"` is invalid.

    2. Dense Label

    Dense label is a representation where one character corresponds to one register.
    For example,

    .. code-block:: python

        "X"
        "IIYYZ-IX++"

    are possible labels.
    Note that dense label cannot represent all of SpinOp.

    **Initialization**

    The :class:`SpinOp` can be initialized by the list of tuples.
    For example,

    .. code-block:: python

        x = SpinOp("X_0", spin=3/2)
        y = SpinOp("Y_0", spin=3/2)
        z = SpinOp("Z_0", spin=3/2)

    are :math:`S_x, S_y, S_z` for spin 3/2 system.
    Two qutrit Heisenberg model with transverse magnetic field is

    .. code-block:: python

        SpinOp(
            [
                ("X_1 X_0", -1),
                ("Y_1 Y_0", -1),
                ("Z_1 Z_0", -1),
                ("Z_1", -0.3),
                ("Z_0", -0.3),
            ],
            spin=1
        )

    This means :math:`- X_1 X_0 - Y_1 Y_0 - Z_1 Z_0 - 0.3 Z_0 - 0.3 Z_1`.

    **Algebra**

    :class:`SpinOp` supports the following basic arithmetic operations: addition, subtraction,
    scalar multiplication, and dagger(adjoint).
    For example,

    .. jupyter-execute::

        from qiskit_nature.operators import SpinOp

        x = SpinOp("X_0", spin=3/2)
        y = SpinOp("Y_0", spin=3/2)
        z = SpinOp("Z_0", spin=3/2)

        print("Raising operator:")
        print(x + 1j * y)
        print("Lowering operator:")
        print(x - 1j * y)

        print("Dagger")
        print(~(1j * z))

    """

    def __init__(
            self,
            data: Union[
                str,
                List[Tuple[str, complex]],
                Tuple[np.ndarray, np.ndarray],
            ],
            spin: float = 1 / 2,
            label_mode: str = "sparse",
    ):
        r"""Initialize ``SpinOp``.

        Args:
            data: label string or list of labels and coefficients. See documentation of SpinOp for
                  more details.
            spin: positive integer or half-integer which represents spin.
            label_mode: The mode of label. (`sparse` (default) or `dense`)

        Raises:
            QiskitNatureError: invalid data is given.
        """
        # 1. Parse input
        if (round(2 * spin) != 2 * spin) or (spin <= 0):
            raise QiskitNatureError("spin must be a positive integer or half-integer")
        self._dim = int(round(2 * spin)) + 1

        # TODO: validation
        # for elem in operator_list:
        #    assert isinstance(elem, SpinOperator)
        #    assert len(elem) == self._register_length, \
        #        'Cannot sum operators acting on registers of different length'
        #    assert elem.spin == self.spin, \
        #        'Cannot sum operators with different spins.'
        if isinstance(data, tuple):
            self._spin_array = np.array(data[0], dtype=np.uint8)
            self._register_length = self._spin_array.shape[1] // 3
            self._coeffs = np.array(data[1], dtype=np.complex128)

        if isinstance(data, str):
            data = [(data, 1)]

        if isinstance(data, list):
            data = self._parse_ladder(data)
            labels, coeffs = zip(*data)
            self._coeffs = np.array(coeffs, np.complex128)
            if label_mode == "sparse":
                self._parse_sparse_label(labels)
            elif label_mode == "dense":
                self._parse_dense_label(labels)

    @property
    def register_length(self):
        return self._register_length

    @property
    def spin(self) -> float:
        """The spin number.

        Returns:
            Spin number
        """
        return (self._dim - 1) / 2

    @property
    def x(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) X operators on the spin system.
        I.e. [0, 4, 2] corresponds to X0^0 \\otimes X1^4 \\otimes X2^2, where Xi acts on the i-th
        spin system in the register.
        """
        return self._spin_array[:, 0: self.register_length]

    @property
    def y(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Y operators on the spin system.
        I.e. [0, 4, 2] corresponds to Y0^0 \\otimes Y1^4 \\otimes Y2^2, where Yi acts on the i-th
        spin system in the register.
        """
        reg_len = self.register_length
        return self._spin_array[:, reg_len: 2 * reg_len]

    @property
    def z(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Z operators on the spin system.
        I.e. [0, 4, 2] corresponds to Z0^0 \\otimes Z1^4 \\otimes Z2^2, where Zi acts on the i-th
        spin system in the register.
        """
        reg_len = self.register_length
        return self._spin_array[:, 2 * reg_len: 3 * reg_len]

    def __repr__(self) -> str:
        # if len(self) == 1 and self.coeff:
        #    return f"SpinOp('{self._labels[0]}')"
        return f"SpinOp({self.to_list()})"  # TODO truncate

    def __str__(self) -> str:
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join([f"{label} * {coeff}" for label, coeff in self.to_list()])

    def add(self, other: "SpinOp") -> "SpinOp":
        if not isinstance(other, SpinOp):
            raise TypeError(
                "Unsupported operand type(s) for +: 'SpinOp' and " f"'{type(other).__name__}'"
            )

        if self.register_length != other.register_length:
            raise TypeError("Incompatible register lengths for '+'.")

        if self.spin != other.spin:
            raise TypeError(f"Addition between spin {self.spin} and spin {other.spin} is invalid.")

        spin_array = np.vstack([self._spin_array, other._spin_array])
        coeffs = self._coeffs + other._coeffs

        return SpinOp((spin_array, coeffs), spin=self.spin)

    def compose(self, other):
        raise NotImplementedError

    def mul(self, other: complex):
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                "Unsupported operand type(s) for *: 'SpinOp' and " f"'{type(other).__name__}'"
            )

        return SpinOp((self._spin_array, self._coeffs * other), spin=self.spin)

    def adjoint(self):
        # Note: X, Y, Z are hermitian, therefore the dagger operation on a SpinOperator amounts
        # to simply complex conjugating the coefficient.
        coeffs = [coeff.conjugate() for coeff in self._coeffs]
        return SpinOp((self._spin_array, coeffs), spin=self.spin)

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "SpinOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        spin_array, indexes = np.unique(self._spin_array, return_inverse=True, axis=0)
        coeff_list = np.zeros(len(self._coeffs), dtype=complex)
        for i, val in zip(indexes, self._coeffs):
            coeff_list[i] += val
        non_zero = [
            i for i, v in enumerate(coeff_list) if not np.isclose(v, 0, atol=atol, rtol=rtol)
        ]
        spin_array = spin_array[non_zero]
        coeff_list = coeff_list[non_zero]
        if not non_zero:
            return SpinOp((
                np.zeros((1, self.register_length * 3), dtype=np.int8),
                np.array([0], dtype=np.complex128)
            ))
        return SpinOp((spin_array, coeff_list))

    def __len__(self):
        return len(self._coeffs)

    def to_list(self) -> List[Tuple[str, complex]]:
        """Getter for the list which represents `self`

        Returns:
            The list [(label, coeff)]
        """
        coeff_list = self._coeffs.tolist()
        return [(self._generate_label(i), coeff_list[i]) for i in range(len(self))]

    def _generate_label(self, i):
        """Generates the string description of `self`."""
        labels_list = []
        for pos, (n_x, n_y, n_z) in enumerate(zip(self.x[i], self.y[i], self.z[i])):
            rev_pos = self.register_length - pos - 1
            if n_x > 1:
                labels_list.append(f"X_{rev_pos}^{n_x}")
            elif n_x == 1:
                labels_list.append(f"X_{rev_pos}")
            if n_y > 1:
                labels_list.append(f"Y_{rev_pos}^{n_y}")
            elif n_y == 1:
                labels_list.append(f"Y_{rev_pos}")
            if n_z > 1:
                labels_list.append(f"Z_{rev_pos}^{n_z}")
            elif n_z == 1:
                labels_list.append(f"Z_{rev_pos}")
            if n_x == n_y == n_z == 0:
                labels_list.append(f"I_{rev_pos}")
        return " ".join(labels_list) if labels_list else f"I_{self.register_length - 1}"

    def to_matrix(self) -> np.ndarray:
        """Convert to dense matrix

        Returns:
            The matrix (numpy.ndarray with dtype=numpy.complex128)
        """
        return cast(
            np.ndarray,
            np.sum(
                [
                    c * self._to_matrix_from_spin_array(self._spin_array[i])
                    for i, c in enumerate(self._coeffs)
                ],
                axis=0,
            )
        )

    def _to_matrix_from_spin_array(self, array):
        reg_len = self.register_length
        x_arr = array[0:reg_len]
        y_arr = array[reg_len: 2 * reg_len]
        z_arr = array[2 * reg_len: 3 * reg_len]
        return reduce(
            np.kron,
            (self._xyz_mat(x, y, z) for x, y, z in zip(x_arr, y_arr, z_arr)),
        )

    def _xyz_mat(self, x, y, z):
        if x == y == z == 0:
            return np.eye(self._dim)

        exist = False
        if x > 0:
            exist = True
            mat = self._x_mat() ** x
        if y > 0:
            mat = mat @ self._y_mat() ** y if exist else self._y_mat()
        if z > 0:
            mat = mat @ self._z_mat() ** z if exist else self._z_mat()
        return mat

    @lru_cache(maxsize=1)
    def _x_mat(self):
        return np.array(
            [
                [
                    np.sqrt((self.spin + 1) * (i + j + 1) - (i + 1) * (j + 1)) / 2
                    if abs(i - j) == 1
                    else 0
                    for j in range(self._dim)
                ]
                for i in range(self._dim)
            ],
            dtype=np.complex128,
        )

    @lru_cache(maxsize=1)
    def _y_mat(self):
        return np.array(
            [
                [
                    1j * (i - j) * np.sqrt((self.spin + 1) * (i + j + 1) - (i + 1) * (j + 1)) / 2
                    if abs(i - j) == 1
                    else 0
                    for j in range(self._dim)
                ]
                for i in range(self._dim)
            ],
            dtype=np.complex128,
        )

    @lru_cache(maxsize=1)
    def _z_mat(self):
        return np.array(
            [
                [(self.spin - i) if i == j else 0 for j in range(self._dim)]
                for i in range(self._dim)
            ],
            dtype=np.complex128,
        )

    def _parse_sparse_label(self, labels):
        xyz_dict = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        re_index = re.compile(r"(?<=[IXYZ]_)\d+((?=\^)|$)")
        num_terms = len(labels)
        parsed_data = []
        max_index = 0
        for label in labels:
            parsed_data_term = []
            label_list = label.split()
            for single in label_list:
                if single[0] not in xyz_dict:
                    raise ValueError(f"Given label {single} must be X, Y, Z, or I.")
                xyz_num = xyz_dict[single[0]]
                match_index = re_index.search(single)
                if match_index:
                    index = int(match_index.group())
                else:
                    raise ValueError(f"Given label {single} has no index.")
                if len(single) == match_index.end():
                    power = 1
                elif (
                        single[match_index.end()] == "^"
                        and single[match_index.end() + 1:].isdecimal()
                ):
                    power = int(single[match_index.end() + 1:])
                else:
                    raise ValueError(f"Invalid label: {single}.")

                max_index = max(max_index, index)

                if xyz_num != 0:
                    parsed_data_term.append((xyz_num, index, power))
            parsed_data.append(parsed_data_term)

        self._register_length = max_index + 1
        self._spin_array = np.zeros((num_terms, self._register_length * 3), dtype=np.uint8)
        for i, data in enumerate(parsed_data):
            for datum in data:
                if (
                        datum[0] == 1
                        and self._spin_array[i][2 * self._register_length - datum[1] - 1] > 0
                ):
                    raise ValueError("Label must be XYZ order.")
                if (
                        datum[0] == 1
                        and self._spin_array[i][3 * self._register_length - datum[1] - 1] > 0
                ):
                    raise ValueError("Label must be XYZ order.")
                if (
                        datum[0] == 2
                        and self._spin_array[i][3 * self._register_length - datum[1] - 1] > 0
                ):
                    raise ValueError("Label must be XYZ order.")

                if self._spin_array[i][datum[0] * self._register_length - datum[1] - 1] != 0:
                    raise ValueError("Duplicate label.")
                self._spin_array[i][datum[0] * self._register_length - datum[1] - 1] = datum[2]

    def _parse_dense_label(self, labels):
        self._register_length = len(labels[0])
        num_terms = len(labels)
        self._spin_array = np.zeros((num_terms, self._register_length * 3), dtype=np.uint8)
        for i, label in enumerate(labels):
            for pos, char in enumerate(label):
                if char == "X":
                    self._spin_array[i][pos] = 1
                elif char == "Y":
                    self._spin_array[i][self._register_length + pos] = 1
                elif char == "Z":
                    self._spin_array[i][2 * self._register_length + pos] = 1

    def _parse_ladder(self, data):
        allowed_str = 'XYZI_^+-0123456789 '
        pattern_plus = re.compile(r'\+')
        pattern_minus = re.compile(r'-')
        new_data = []
        for label, coeff in data:
            if not all(char in allowed_str for char in label):
                raise ValueError(
                    f"Invalid label: {label}."
                    "Label must consist of X, Y, Z, I , +, -, ^, _, 0-9, and spaces."
                )
            plus_indices = [m.start() for m in pattern_plus.finditer(label)]
            minus_indices = [m.start() for m in pattern_minus.finditer(label)]
            len_plus = len(plus_indices)
            len_minus = len(minus_indices)
            pm_indices = plus_indices + minus_indices
            label_list = list(label)
            for indices in product(["X", "Y"], repeat=len_plus+len_minus):
                for i, index in enumerate(indices):
                    label_list[pm_indices[i]] = index
                phase = indices[:len_plus].count("Y") - indices[len_plus:].count("Y")
                new_data.append((''.join(label_list), coeff * 1j ** phase))

        return new_data
