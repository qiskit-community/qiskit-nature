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

"""Test for Tensor class"""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase

import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.operators import FermionicOp, PolynomialTensor, Tensor
from qiskit_nature.second_q.operators.tensor_ordering import (
    IndexType,
    find_index_order,
    to_physicist_ordering,
)

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
    """Test a dense Tensor."""

    tensor = Tensor(np.arange(1, 17, dtype=float).reshape((4, 4)))
    expected = np.arange(1, 17, dtype=float).reshape((4, 4))


@unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
class TestSparseTensor(QiskitNatureTestCase, MatrixTensorTestCase):
    """Test a sparse Tensor."""

    def setUp(self) -> None:
        super().setUp()
        # pylint: disable=import-error
        import sparse as sp

        self.expected = sp.DOK.from_numpy(np.arange(1, 17, dtype=float).reshape((4, 4)))
        self.tensor = Tensor(self.expected)


class TestChemOrdered2Body(QiskitNatureTestCase):
    """Tests the ``Tensor.label_template`` mechanics."""

    def test_chem_eri(self):
        """This test mirrors the example from the docstring of that attribute and, thus, ensures
        that chemistry-ordered 2-body electronic integrals do in fact work as advertised.
        """
        chem_eri = np.asarray(
            [
                [
                    [[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                ],
                [
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                    [[0.57187698, 0.44744572], [0.44744572, 0.77460594]],
                ],
            ]
        )
        self.assertEqual(find_index_order(chem_eri), IndexType.CHEMIST)

        chem_tensor = Tensor(chem_eri)
        chem_tensor.label_template = "+_{{0}} +_{{2}} -_{{3}} -_{{1}}"

        chem_op = FermionicOp.from_polynomial_tensor(PolynomialTensor({"++--": chem_tensor}))

        phys_eri = to_physicist_ordering(chem_eri)

        phys_op = FermionicOp.from_polynomial_tensor(PolynomialTensor({"++--": phys_eri}))

        self.assertTrue(chem_op.equiv(phys_op))


if __name__ == "__main__":
    unittest.main()
