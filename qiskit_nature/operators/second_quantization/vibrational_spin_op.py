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
"""A Vibrational Spin operator.
Note: this implementation differs fundamentally from the `FermionicOperator` and `BosonicOperator`
as it relies an the mathematical representation of spin matrices as (e.g.) explained in [1].
[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""
import re
from fractions import Fraction
from functools import lru_cache, partial, reduce
from typing import Optional, List, Tuple, Union, cast

import numpy as np

from .particle_op import ParticleOp
from ... import QiskitNatureError


# TODO decide whether it inherits from SpinOp, if yes, some methods will go away from here
class VibrationalSpinOp(ParticleOp):
    """Vibrational Spin type operators.
    **Label**
    Allowed characters for primitives of labels are + and -.
    .. list-table::
        :header-rows: 1
        * - `+`
          - :math:`S_+`
          - Raising operator
        * - `-`
          - :math:`S_-`
          - Lowering operator
    1. Sparse Label (if underscore `_` exists in the label)
    # TODO explain labelling strategy when decided
    **Initialization**
    # TODO
    **Algebra**
    :class:`VibrationalSpinOp` supports the following basic arithmetic operations: addition,
    subtraction,
    scalar multiplication, and dagger(adjoint).
    For example,
    .. jupyter-execute::
        from qiskit_nature.operators import VibrationalSpinOp
        print("Raising operator:")
        print(x + 1j * y)
        plus = SpinOp("+", spin=3/2)
        print("This is same with: ", plus)
        print("Lowering operator:")
        print(x - 1j * y)
        minus = SpinOp("-", spin=3/2)
        print("This is same with: ", minus)
        print("Dagger")
        print(~(1j * z))
    """

    _XYZ_DICT = {"X": 0, "Y": 1, "Z": 2}

    _VALID_LABEL_PATTERN = re.compile(
        r"^([IXYZ\+\-]_\d(\^\d)?\s)*[IXYZ\+\-]_\d(\^\d)?(?!\s)$|^[IXYZ\+\-]+$")
    # TODO do we want XYZ or only +-?
    _SPARSE_LABEL_PATTERN = re.compile(r"^([IXYZ]_\d\s)*[+-]_\d(?!\s)$")

    def __init__(
            self,
            data: Union[
                str,
                List[Tuple[str, complex]],
                Tuple[np.ndarray, np.ndarray],
            ],
            num_modes: int, num_modals: Union[int, List[int]],
            spin: Union[float, Fraction] = Fraction(1, 2),

    ):
        r"""
        Args:
            data: label string, list of labels and coefficients. See the label section in
                  the documentation of :class:`VibrationalSpinOp` for more details.
            num_modes : number of modes
            num_modals: number of modals
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
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

        self.num_modals = num_modals
        self.num_modes = num_modes

        # TODO validate num_modals vs num_modes

        if isinstance(data, tuple):
            self._spin_array = np.array(data[0], dtype=np.uint8)
            self._register_length = self._spin_array.shape[2]
            self._coeffs = np.array(data[1], dtype=dtype)

        if isinstance(data, str):
            data = [(data, 1)]

        if isinstance(data, list):
            invalid_labels = [label for label, _ in data if self._VALID_LABEL_PATTERN.match(label)]
            if not invalid_labels:
                raise ValueError(f"Invalid labels: {invalid_labels}")

            data = self._flatten_ladder_ops(data)

            labels, coeffs = zip(*data)
            self._coeffs = np.array(coeffs, dtype=dtype)

            if all(self._SPARSE_LABEL_PATTERN.match(label) for label in labels):
                self._from_sparse_label(labels)
            else:
                raise ValueError(
                    f"Mixed labels are included in {labels}. "
                    "You can only use either of spare or dense label"
                )
        # Make immutable
        self._spin_array.flags.writeable = False
        self._coeffs.flags.writeable = False

    def __repr__(self) -> str:
        if len(self) == 1 and self._coeffs[0] == 1:
            return f"SpinOp('{self.to_list()[0][0]}')"
        return f"SpinOp({self.to_list()}, spin={self.spin})"  # TODO truncate

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
        I.e. [0, 4, 2] corresponds to X_2^0 \\otimes X_1^4 \\otimes X_0^2, where X_i acts on the
        i-th spin system in the register.
        """
        return self._spin_array[0]

    @property
    def y(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Y operators on the spin system.
        I.e. [0, 4, 2] corresponds to Y_2^0 \\otimes Y_1^4 \\otimes Y_0^2, where Y_i acts on the
        i-th spin system in the register.
        """
        return self._spin_array[1]

    @property
    def z(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Z operators on the spin system.
        I.e. [0, 4, 2] corresponds to Z_2^0 \\otimes Z_1^4 \\otimes Z_0^2, where Z_i acts on the
        i-th spin system in the register.
        """
        return self._spin_array[2]

    def _is_num_modals_valid(self):
        if type(self.num_modals) == list and len(self.num_modals) != self.num_modes:
            return False
        return True

    def add(self, other):
        raise NotImplementedError()

    def compose(self, other):
        # TODO: implement
        raise NotImplementedError()

    def mul(self, other: complex):
        raise NotImplementedError()

    def adjoint(self):
        raise NotImplementedError()

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None):
        raise NotImplementedError()

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
            rev_pos = self.register_length - pos - 1
            if n_x == n_y == n_z == 0:
                labels_list.append(f"I_{rev_pos}")
                continue
            if n_x >= 1:
                labels_list.append(f"X_{rev_pos}" + (f"^{n_x}" if n_x > 1 else ""))
            if n_y >= 1:
                labels_list.append(f"Y_{rev_pos}" + (f"^{n_y}" if n_y > 1 else ""))
            if n_z >= 1:
                labels_list.append(f"Z_{rev_pos}" + (f"^{n_z}" if n_z > 1 else ""))
        return " ".join(labels_list)

    @lru_cache()
    def to_matrix(self) -> np.ndarray:
        """Convert to dense matrix
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
        return mat

    # TODO change logic to new labels
    def _from_sparse_label(self, labels):
        num_terms = len(labels)
        parsed_data = []
        max_index = 0
        for term, label in enumerate(labels):
            label_list = label.split()
            for single_label in label_list:
                xyz, nums = single_label.split("_", 1)
                index_str, power_str = nums.split("^", 1) if "^" in nums else (nums, "1")

                index = int(index_str)
                power = int(power_str)
                max_index = max(max_index, index)

                if xyz in self._XYZ_DICT:
                    parsed_data.append((term, self._XYZ_DICT[xyz], index, power))

        self._register_length = max_index + 1
        self._spin_array = np.zeros((3, num_terms, self._register_length), dtype=np.uint8)
        for term, xyz_num, index, power in parsed_data:
            register = self._register_length - index - 1

            # Check the order of X, Y, and Z whether it has been already assigned.
            if self._spin_array[range(xyz_num + 1, 3), term, register].any():
                raise ValueError("Label must be XYZ order.")
            # same label is not assigned.
            if self._spin_array[xyz_num, term, register]:
                raise ValueError("Duplicate label.")

            self._spin_array[xyz_num, term, register] = power

    # TODO change logic to new labels
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
            for indices in np.product(["X", "Y"], repeat=len_plus + len_minus):
                for i, index in enumerate(indices):
                    label_list[pm_indices[i]] = index
                # The phase is determined by the number of Y in + and - respectively. For example,
                # S_+ otimes S_- = (X + i Y) otimes (X - i Y)
                # = i^{0-0} XX + i^{1-0} YX + i^{0-1} XY + i^{1-1} YY
                # = XX + i YX - i XY + YY
                phase = indices[:len_plus].count("Y") - indices[len_plus:].count("Y")
                new_data.append(("".join(label_list), coeff * 1j ** phase))

        return new_data
