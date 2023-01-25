# This code is part of Qiskit.
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

"""Test for Tensor class"""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase

import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.operators import Tensor

from .tensor_test_cases import MatrixTensorTestCase, ScalarTensorTestCase


class TestIntTensor(QiskitNatureTestCase, ScalarTensorTestCase):
    """Test numeric Tensor with integer value."""

    tensor = Tensor(1)
    expected = np.array(1)


class TestFloatTensor(QiskitNatureTestCase, ScalarTensorTestCase):
    """Test numeric Tensor with float value."""

    tensor = Tensor(1.0)
    expected = np.array(1.0)


class TestComplexTensor(QiskitNatureTestCase, ScalarTensorTestCase):
    """Test numeric Tensor with complex value."""

    tensor = Tensor(1.0j)
    expected = np.array(1.0j)


class TestNumpyIntTensor(QiskitNatureTestCase, ScalarTensorTestCase):
    """Test numeric Tensor with numpy integer value."""

    tensor = Tensor(np.int64(1))
    expected = np.array(np.int64(1))


class TestNumpyFloatTensor(QiskitNatureTestCase, ScalarTensorTestCase):
    """Test numeric Tensor with numpy float value."""

    tensor = Tensor(np.float64(1.0))
    expected = np.array(np.float64(1.0))


class TestNumpyComplexTensor(QiskitNatureTestCase, ScalarTensorTestCase):
    """Test numeric Tensor with numpy complex value."""

    tensor = Tensor(np.complex128(1.0j))
    expected = np.array(np.complex128(1.0j))


class TestNumpyTensor(QiskitNatureTestCase, MatrixTensorTestCase):
    """TODO."""

    tensor = Tensor(np.arange(1, 17, dtype=float).reshape((4, 4)))
    expected = np.arange(1, 17, dtype=float).reshape((4, 4))


@unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
class TestSparseTensor(QiskitNatureTestCase, MatrixTensorTestCase):
    """TODO."""

    def setUp(self) -> None:
        """TODO."""
        super().setUp()
        # pylint: disable=import-error
        import sparse as sp

        self.expected = sp.DOK.from_numpy(np.arange(1, 17, dtype=float).reshape((4, 4)))
        self.tensor = Tensor(self.expected)


if __name__ == "__main__":
    unittest.main()
