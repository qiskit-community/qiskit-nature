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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the `FermionicOp`
as it relies on the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

import re
from fractions import Fraction
from functools import lru_cache, partial, reduce
from itertools import product
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from qiskit.utils.validation import validate_min
from qiskit_nature import QiskitNatureError
from qiskit_nature.deprecation import deprecate_function

from .second_quantized_op import SecondQuantizedOp


class SpinOp(SecondQuantizedOp):
    """XYZ-ordered Spin operators.

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
          - :math:`S^x`
          - :math:`x`-component of the spin operator
        * - `Y`
          - :math:`S^y`
          - :math:`y`-component of the spin operator
        * - `Z`
          - :math:`S^z`
          - :math:`z`-component of the spin operator
        * - `+`
          - :math:`S^+`
          - Raising operator
        * - `-`
          - :math:`S^-`
          - Lowering operator

    There are two types of label modes for :class:`SpinOp`.
    The label mode is automatically detected.

    1. Dense Label (default, `register_length = None`)

    Dense labels are strings in which each character maps to a unique spin mode.
    This is similar to Qiskit's string-based representation of qubit operators.
    For example,

    .. code-block:: python

        "X"
        "IIYYZ-IX++"

    are possible labels.
    Note, that dense labels are less powerful than sparse ones because they cannot represent
    all possible :class:`SpinOp`. You will, for example, not be able to apply multiple operators
    on the same index within a single label.

    2. Sparse Label (`register_length` is passed)

    A sparse label is a string consisting of a space-separated list of words.
    Each word must look like :code:`[XYZI+-]_<index>^<power>`,
    where the :code:`<index>` is a non-negative integer representing the index of the spin mode
    and the :code:`<power>` is a positive integer indicating the number of times the given operator
    is applied to the mode at :code:`<index>`.
    You can omit :code:`<power>`, implying a single application of the operator (:code:`power = 1`).
    For example,

    .. code-block:: python

        "X_0"
        "Y_0^2"
        "Y_0^2 Z_0^3 X_1^1 Y_1^2 Z_1^2"

    are possible labels.
    For each :code:`index` the operations `X`, `Y` and `Z` can only be specified exclusively in
    this order. `+` and `-` cannot be used with `X` and `Y`
    because ladder operators will be parsed into `X` and `Y`.
    Thus, :code:`"Z_0 X_0"`, :code:`"Z_0 +_0"`, and :code:`"+_0 X_0"` are invalid labels.
    The indices must be ascending order.

    :code:`"+_i -_i"` is supported.
    This pattern is parsed to :code:`+_i -_i = X_i^2 + Y_i^2 + Z_i`.

    **Initialization**

    The :class:`SpinOp` can be initialized by the list of tuples.
    For example,

    .. jupyter-execute::

        from qiskit_nature.second_q.operators import SpinOp

        x = SpinOp("X", spin=3/2)
        y = SpinOp("Y", spin=3/2)
        z = SpinOp("Z", spin=3/2)

    are :math:`S^x, S^y, S^z` for spin 3/2 system.
    Two qutrit Heisenberg model with transverse magnetic field is

    .. jupyter-execute::

        SpinOp(
            [
                ("XX", -1),
                ("YY", -1),
                ("ZZ", -1),
                ("ZI", -0.3),
                ("IZ", -0.3),
            ],
            spin=1
        )

    This means :math:`- S^x_0 S^x_1 - S^y_0 S^y_1 - S^z_0 S^z_1 - 0.3 S^z_0 - 0.3 S^z_1`.

    :class:`SpinOp` can be initialized with internal data structure (`numpy.ndarray`) directly.
    In this case, `data` is a tuple of two elements: `spin_array` and `coeffs`.
    `spin_array` is 3-dimensional `ndarray`. 1st axis has three elements 0, 1, and 2 corresponding
    to x, y, and z.  2nd axis represents the index of terms.
    3rd axis represents the index of register.
    `coeffs` is one-dimensional `ndarray` with the length of the number of terms.

    **Algebra**

    :class:`SpinOp` supports the following basic arithmetic operations: addition, subtraction,
    scalar multiplication, and adjoint.
    For example,

    Raising Operator (addition and scalar multiplication)

    .. jupyter-execute::

        x + 1j * y

    Adjoint

    .. jupyter-execute::

        ~(1j * z)

    """

    def __init__(
        self,
        data: Union[
            str,
            List[Tuple[str, complex]],
            Tuple[np.ndarray, np.ndarray],
        ],
        spin: Union[float, Fraction] = Fraction(1, 2),
        register_length: Optional[int] = None,
    ):
        r"""
        Args:
            data: label string, list of labels and coefficients. See the label section in
                  the documentation of :class:`SpinOp` for more details.
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
            register_length: length of the particle register.

        Raises:
            ValueError: invalid data is given.
            QiskitNatureError: invalid spin value
        """
        self._coeffs: np.ndarray
        self._spin_array: np.ndarray
        dtype = np.complex128  # TODO: configurable data type. mixin?

        spin = Fraction(spin)
        if spin.denominator not in (1, 2):
            raise QiskitNatureError(
                f"spin must be a positive half-integer (integer or half-odd-integer), not {spin}."
            )
        self._dim = int(2 * spin + 1)

        if isinstance(data, tuple) and all(isinstance(datum, np.ndarray) for datum in data):
            self._spin_array = np.array(data[0], dtype=np.uint8)
            self._register_length = self._spin_array.shape[2]
            self._coeffs = np.array(data[1], dtype=dtype)

        if (
            isinstance(data, tuple)
            and isinstance(data[0], str)
            and isinstance(data[1], (int, float, complex))
        ):
            data = [data]

        if isinstance(data, str):
            data = [(data, 1)]

        if isinstance(data, list):
            if register_length is not None:  # Sparse label
                # [IXYZ]_index^power (power is optional) or [+-]_index
                sparse = r"([IXYZ]_\d+(\^\d+)?|[\+\-]_\d+?)"
                # space (\s) separated sparse label or empty string
                label_pattern = re.compile(rf"^({sparse}\s)*{sparse}(?!\s)$|^$")
                invalid_labels = [label for label, _ in data if not label_pattern.match(label)]
                if invalid_labels:
                    raise ValueError(f"Invalid labels for sparse labels: {invalid_labels}.")
            else:  # dense_label
                # dense label (repeat of [IXYZ+-])
                label_pattern = re.compile(r"^[IXYZ\+\-]+$")
                invalid_labels = [label for label, _ in data if not label_pattern.match(label)]
                if invalid_labels:
                    raise ValueError(
                        f"Invalid labels for dense labels: {invalid_labels} (if you want to use "
                        "sparse label, you forgot a parameter `register_length`.)"
                    )

            # Parse ladder operators for special patterns.
            if register_length is not None:
                data = self._flatten_raising_lowering_ops(data, register_length)
            data = self._flatten_ladder_ops(data)

            # set coeffs
            labels, coeffs = zip(*data)
            self._coeffs = np.array(coeffs, dtype=dtype)

            # set labels
            if register_length is None:  # Dense label
                self._register_length = len(labels[0])
                label_pattern = re.compile(r"^[IXYZ]+$")
                invalid_labels = [label for label in labels if not label_pattern.match(label)]
                if invalid_labels:
                    raise ValueError(f"Invalid labels for dense labels are given: {invalid_labels}")
                self._spin_array = np.array(
                    [
                        [[char == "X", char == "Y", char == "Z"] for char in label]
                        for label in labels
                    ],
                    dtype=np.uint8,
                ).transpose((2, 0, 1))
            else:  # Sparse label
                validate_min("register_length", register_length, 1)
                label_pattern = re.compile(r"^[IXYZ]_\d+(\^\d+)?$")
                invalid_labels = [
                    label
                    for label in labels
                    if not all(label_pattern.match(lb) for lb in label.split())
                ]
                if invalid_labels:
                    raise ValueError(
                        f"Invalid labels for sparse labels are given: {invalid_labels}"
                    )
                self._register_length = register_length
                self._from_sparse_label(labels)

        # Make immutable
        self._spin_array.flags.writeable = False
        self._coeffs.flags.writeable = False

    def __repr__(self) -> str:
        spin = self.spin
        reg_len = self.register_length
        if len(self) == 1:
            if self._coeffs[0] == 1:  # str
                data_str = f"'{self.to_list()[0][0]}'"
            else:  # tuple
                data_str = repr(self.to_list()[0])
        else:  # list
            data_str = repr(self.to_list())
        return f"SpinOp({data_str}, spin={spin}, register_length={reg_len})"  # TODO truncate

    def __str__(self) -> str:
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join([f"{label} * {coeff}" for label, coeff in self.to_list()])

    def __len__(self) -> int:
        return len(self._coeffs)

    @property
    def register_length(self):
        return self._register_length

    @property
    def spin(self) -> Fraction:
        """The spin number.

        Returns:
            Spin number
        """
        return Fraction(self._dim - 1, 2)

    @property
    def x(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) X operators on the spin system.
        I.e. [0, 4, 2] corresponds to X_0^0 \\otimes X_1^4 \\otimes X_2^2, where X_i acts on the
        i-th spin system in the register.
        """
        return self._spin_array[0]

    @property
    def y(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Y operators on the spin system.
        I.e. [0, 4, 2] corresponds to Y_0^0 \\otimes Y_1^4 \\otimes Y_2^2, where Y_i acts on the
        i-th spin system in the register.
        """
        return self._spin_array[1]

    @property
    def z(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Z operators on the spin system.
        I.e. [0, 4, 2] corresponds to Z_0^0 \\otimes Z_1^4 \\otimes Z_2^2, where Z_i acts on the
        i-th spin system in the register.
        """
        return self._spin_array[2]

    def add(self, other: "SpinOp") -> "SpinOp":
        if not isinstance(other, SpinOp):
            raise TypeError(
                "Unsupported operand type(s) for +: 'SpinOp' and " f"'{type(other).__name__}'"
            )

        if self.register_length != other.register_length:
            raise TypeError("Incompatible register lengths for '+'.")

        if self.spin != other.spin:
            raise TypeError(f"Addition between spin {self.spin} and spin {other.spin} is invalid.")

        return SpinOp(
            (
                np.hstack((self._spin_array, other._spin_array)),
                np.hstack((self._coeffs, other._coeffs)),
            ),
            spin=self.spin,
        )

    def compose(self, other):
        # TODO: implement
        raise NotImplementedError

    def mul(self, other: complex) -> "SpinOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SpinOp' and '{type(other).__name__}'"
            )

        return SpinOp((self._spin_array, self._coeffs * other), spin=self.spin)

    def adjoint(self) -> "SpinOp":
        if (self._spin_array.sum(axis=0) > 1).any():
            # TODO: implement this when compose() will be implemented.
            raise NotImplementedError(
                "Adjoint for an operator which have multiple operators for the same register."
            )
        # Note: X, Y, Z are hermitian, therefore the dagger operation on a SpinOperator amounts
        # to simply complex conjugating the coefficient.
        return SpinOp((self._spin_array, self._coeffs.conjugate()), spin=self.spin)

    @deprecate_function("0.4.0", new_name="simplify")
    def reduce(
        self,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,  # pylint: disable=unused-argument
    ) -> "SpinOp":
        """Reduce the operator.

        This method is deprecated. Use `simplify` instead.
        """
        return self.simplify(atol=atol)

    def simplify(self, atol: Optional[float] = None) -> "SpinOp":
        """Simplify the operator.

        Merges terms with same labels and eliminates terms with coefficients close to 0.
        Returns a new operator (the original operator is not modified).

        Args:
            atol: Absolute tolerance for checking if coefficients are zero (Default: 1e-8).

        Returns:
            The simplified operator.
        """
        if atol is None:
            atol = self.atol

        flatten_array, indices = np.unique(
            np.column_stack(cast(Sequence, self._spin_array)),
            return_inverse=True,
            axis=0,
        )
        coeff_list = np.zeros(flatten_array.shape[0], dtype=np.complex128)
        np.add.at(coeff_list, indices, self._coeffs)
        is_zero = np.isclose(coeff_list, 0, atol=atol)
        if np.all(is_zero):
            return SpinOp(
                (
                    np.zeros((3, 1, self.register_length), dtype=np.int8),
                    np.array([0], dtype=np.complex128),
                ),
                spin=self.spin,
            )
        non_zero = np.logical_not(is_zero)
        new_array = (
            flatten_array[non_zero]
            .reshape((np.count_nonzero(non_zero), 3, self.register_length))
            .transpose(1, 0, 2)
        )
        new_coeff = coeff_list[non_zero]
        return SpinOp((new_array, new_coeff), spin=self.spin)

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
        for pos, (n_x, n_y, n_z) in enumerate(self._spin_array[:, i].T):
            if n_x >= 1:
                labels_list.append(f"X_{pos}" + (f"^{n_x}" if n_x > 1 else ""))
            if n_y >= 1:
                labels_list.append(f"Y_{pos}" + (f"^{n_y}" if n_y > 1 else ""))
            if n_z >= 1:
                labels_list.append(f"Z_{pos}" + (f"^{n_z}" if n_z > 1 else ""))
        if not labels_list:
            return f"I_{self.register_length - 1}"
        return " ".join(labels_list)

    @lru_cache(maxsize=128)
    def to_matrix(self) -> np.ndarray:
        """Convert to dense matrix.

        Returns:
            The matrix (numpy.ndarray with dtype=numpy.complex128)
        """
        # TODO: use scipy.sparse.csr_matrix() and add parameter `sparse: bool`.
        x_mat = np.fromfunction(
            lambda i, j: np.where(
                np.abs(i - j) == 1,
                np.sqrt((self._dim + 1) * (i + j + 1) / 2 - (i + 1) * (j + 1)) / 2,
                0,
            ),
            (self._dim, self._dim),
            dtype=np.complex128,
        )
        y_mat = np.fromfunction(
            lambda i, j: np.where(
                np.abs(i - j) == 1,
                1j * (i - j) * np.sqrt((self._dim + 1) * (i + j + 1) / 2 - (i + 1) * (j + 1)) / 2,
                0,
            ),
            (self._dim, self._dim),
            dtype=np.complex128,
        )
        z_mat = np.fromfunction(
            lambda i, j: np.where(i == j, (self._dim - 2 * i - 1) / 2, 0),
            (self._dim, self._dim),
            dtype=np.complex128,
        )

        tensorall = partial(reduce, np.kron)

        mat = sum(
            self._coeffs[i]
            * tensorall(
                np.linalg.matrix_power(x_mat, x)
                @ np.linalg.matrix_power(y_mat, y)
                @ np.linalg.matrix_power(z_mat, z)
                for x, y, z in self._spin_array[:, i].T
            )
            for i in range(len(self))
        )
        mat = cast(np.ndarray, mat)
        mat.flags.writeable = False
        return mat.view()

    def _from_sparse_label(self, labels):
        xyz_dict = {"X": 0, "Y": 1, "Z": 2}

        # 3-dimensional ndarray (XYZ, terms, register)
        self._spin_array = np.zeros((3, len(labels), self.register_length), dtype=np.uint8)
        for term, label in enumerate(labels):
            for split_label in label.split():
                xyz, nums = split_label.split("_", 1)

                if xyz not in xyz_dict:
                    continue

                xyz_num = xyz_dict[xyz]
                index, power = map(int, nums.split("^", 1)) if "^" in nums else (int(nums), 1)
                if index >= self.register_length:
                    raise ValueError(
                        f"Index {index} must be smaller than register_length {self.register_length}"
                    )
                # Check the order of X, Y, and Z whether it has been already assigned.
                if self._spin_array[range(xyz_num + 1, 3), term, index].any():
                    raise ValueError(f"Label must be in XYZ order, but {label}.")
                # same index is not assigned.
                if self._spin_array[xyz_num, term, index]:
                    raise ValueError(f"Duplicate index label {index} is given.")

                self._spin_array[xyz_num, term, index] = power

    @staticmethod
    def _flatten_ladder_ops(data):
        """Convert `+` to `X + 1j Y` and `-` to `X - 1j Y` with the distributive law"""
        new_data = []
        for label, coeff in data:
            plus_indices = [i for i, char in enumerate(label) if char == "+"]
            minus_indices = [i for i, char in enumerate(label) if char == "-"]
            len_plus = len(plus_indices)
            len_minus = len(minus_indices)
            pm_indices = plus_indices + minus_indices
            label_list = list(label)
            for indices in product(["X", "Y"], repeat=len_plus + len_minus):
                for i, index in enumerate(indices):
                    label_list[pm_indices[i]] = index
                # The phase is determined by the number of Y in + and - respectively. For example,
                # S_+ otimes S_- = (X + i Y) otimes (X - i Y)
                # = i^{0-0} XX + i^{1-0} YX + i^{0-1} XY + i^{1-1} YY
                # = XX + i YX - i XY + YY
                phase = indices[:len_plus].count("Y") - indices[len_plus:].count("Y")
                new_data.append(("".join(label_list), coeff * 1j**phase))

        return new_data

    @staticmethod
    def _flatten_raising_lowering_ops(data, register_length):
        """Convert +_i -_i to X_i^2 + Y_i^2 + Z_i"""
        new_data = []
        for label, coeff in data:
            positions = []
            indices = []
            label_list = label.split()
            for i in range(register_length):
                if f"+_{i}" in label_list and f"-_{i}" in label_list:
                    plus_pos = label_list.index(f"+_{i}")
                    minus_pos = label_list.index(f"-_{i}")
                    if minus_pos - plus_pos == 1:
                        positions.append(plus_pos)
                        indices.append(i)
            for ops in product(*([f"X_{i}^2", f"Y_{i}^2", f"Z_{i}"] for i in indices)):
                label_list = label.split()
                for pos, op in zip(positions, ops):
                    label_list[pos] = op
                for pos in sorted(positions, reverse=True):
                    label_list.pop(pos + 1)
                new_data.append((" ".join(label_list), coeff))
        return new_data
