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
from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.operators.tensor import ARRAY_TYPE, Tensor


class ScalarTensorTestCase(ABC):
    """Tests for Tensor class wrapping a single numeric value."""

    tensor: Tensor
    expected: np.ndarray

    @abstractmethod
    def subTest(self, msg, **kwargs):
        # pylint: disable=invalid-name
        """subtest"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """assert equal"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertNotEqual(self, first, second, msg=None):
        """assert not equal"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertTrue(self, condition, msg=None):
        """assert true"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertFalse(self, condition, msg=None):
        """assert false"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    def test_array(self):
        """Test array property."""
        self.assertTrue(isinstance(self.tensor.array, type(self.expected)))

    def test_shape(self):
        """Test shape property."""
        self.assertEqual(self.tensor.shape, ())

    def test_ndim(self):
        """Test ndim property."""
        self.assertEqual(self.tensor.ndim, 0)

    def test_equality(self):
        """Test equality check."""
        self.assertEqual(self.tensor, Tensor(self.expected))

    def test_equivalence(self):
        """Test equivalence check."""
        self.assertTrue(self.tensor.equiv(Tensor(self.expected + 1e-8)))

        with self.subTest("changing tolerances"):
            self.assertTrue(self.tensor.equiv(Tensor(self.expected + 1e-8)))
            prev_atol = Tensor.atol
            prev_rtol = Tensor.rtol
            Tensor.atol = 1e-10
            Tensor.rtol = 1e-10
            self.assertFalse(self.tensor.equiv(Tensor(self.expected + 1e-8)))
            Tensor.atol = prev_atol
            Tensor.rtol = prev_rtol

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_is_sparse(self):
        """Test sparsity check."""
        self.assertTrue(self.tensor.is_sparse())

    def test_is_dense(self):
        """Test density check."""
        self.assertTrue(self.tensor.is_dense())

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_to_sparse(self):
        """Test to sparse conversion."""
        self.assertEqual(self.tensor.to_sparse(), self.tensor)

    def test_to_dense(self):
        """Test to dense conversion."""
        self.assertEqual(self.tensor.to_dense(), self.tensor)

    def test_add(self):
        """Test addition."""
        self.assertEqual(2 + self.tensor, 2 + self.expected)

    def test_mul(self):
        """Test scalar multiplication."""
        self.assertEqual(2 * self.tensor, 2 * self.expected)

    def test_compose(self):
        """Test composition."""
        a = Tensor(2 * self.tensor)
        b = Tensor(3 * self.tensor)

        with self.subTest("front=False"):
            self.assertEqual(a.compose(b), 6 * self.expected**2)

        with self.subTest("front=True"):
            self.assertEqual(a.compose(b, front=True), 6 * self.expected**2)

    def test_tensor(self):
        """Test tensoring."""
        a = Tensor(2 * self.tensor)
        b = Tensor(3 * self.tensor)

        self.assertEqual(a.tensor(b), 6 * self.expected**2)

    def test_expand(self):
        """Test expanding."""
        a = Tensor(2 * self.tensor)
        b = Tensor(3 * self.tensor)

        self.assertEqual(a.expand(b), 6 * self.expected**2)

    def test_coord_iter(self):
        """Test iterating."""
        self.assertEqual(list(self.tensor.coord_iter()), [(self.expected.item(), tuple())])

    def test_copy(self):
        """Test copying."""
        c = copy(self.tensor)
        self.assertEqual(self.tensor, c)
        self.assertNotEqual(id(self.tensor), id(c))
        self.assertEqual(id(self.tensor.array), id(c.array))

    def test_deepcopy(self):
        """Test deep-copying."""
        c = deepcopy(self.tensor)
        self.assertEqual(self.tensor, c)
        self.assertNotEqual(id(self.tensor), id(c))
        self.assertNotEqual(id(self.tensor.array), id(c.array))


class MatrixTensorTestCase(ABC):
    """Tests for Tensor class wrapping a dense or sparse array."""

    tensor: Tensor
    expected: ARRAY_TYPE

    @abstractmethod
    def subTest(self, msg, **kwargs):
        # pylint: disable=invalid-name
        """subtest"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """assert equal"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertNotEqual(self, first, second, msg=None):
        """assert not equal"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertTrue(self, condition, msg=None):
        """assert true"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    @abstractmethod
    def assertFalse(self, condition, msg=None):
        """assert false"""
        raise Exception("Abstract method")  # pylint: disable=broad-exception-raised

    def test_array(self):
        """Test array property."""
        self.assertTrue(isinstance(self.tensor.array, type(self.expected)))

    def test_shape(self):
        """Test shape property."""
        self.assertEqual(self.tensor.shape, self.expected.shape)

    def test_ndim(self):
        """Test ndim property."""
        self.assertEqual(self.tensor.ndim, self.expected.ndim)

    def test_equality(self):
        """Test equality check."""
        self.assertEqual(self.tensor, Tensor(self.expected))

    def test_equivalence(self):
        """Test equivalence check."""
        slightly_different = deepcopy(self.expected)
        slightly_different[0, 0] += 1e-8
        self.assertTrue(self.tensor.equiv(Tensor(slightly_different)))

        with self.subTest("changing tolerances"):
            self.assertTrue(self.tensor.equiv(Tensor(slightly_different)))
            prev_atol = Tensor.atol
            prev_rtol = Tensor.rtol
            Tensor.atol = 1e-10
            Tensor.rtol = 1e-10
            self.assertFalse(self.tensor.equiv(Tensor(slightly_different)))
            Tensor.atol = prev_atol
            Tensor.rtol = prev_rtol

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_is_sparse(self):
        """Test sparsity check."""
        if isinstance(self.expected, np.ndarray):
            self.assertFalse(self.tensor.is_sparse())
        else:
            self.assertTrue(self.tensor.is_sparse())

    def test_is_dense(self):
        """Test density check."""
        if isinstance(self.expected, np.ndarray):
            self.assertTrue(self.tensor.is_dense())
        else:
            self.assertFalse(self.tensor.is_dense())

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_to_sparse(self):
        """Test to sparse conversion."""
        if isinstance(self.expected, np.ndarray):
            # pylint: disable=import-error
            import sparse as sp

            self.assertEqual(self.tensor.to_sparse(), Tensor(sp.as_coo(self.expected)))
        else:
            self.assertEqual(self.tensor.to_sparse(), self.tensor)

    def test_to_dense(self):
        """Test to dense conversion."""
        if isinstance(self.expected, np.ndarray):
            self.assertEqual(self.tensor.to_dense(), self.tensor)
        else:
            self.assertEqual(self.tensor.to_dense(), Tensor(self.expected.todense()))

    def test_add(self):
        """Test addition."""
        self.assertTrue(Tensor(self.tensor + self.tensor).equiv(Tensor(2 * self.expected)))

    def test_mul(self):
        """Test scalar multiplication."""
        self.assertTrue(Tensor(2 * self.tensor).equiv(Tensor(2 * self.expected)))

    def test_compose(self):
        """Test composition."""
        a = Tensor(2 * self.tensor)
        b = Tensor(3 * self.tensor)

        with self.subTest("front=False"):
            expected = np.outer(2 * self.expected, 3 * self.expected).reshape((4, 4, 4, 4))
            self.assertTrue(a.compose(b).equiv(Tensor(expected)))

        with self.subTest("front=True"):
            expected = np.outer(3 * self.expected, 2 * self.expected).reshape((4, 4, 4, 4))
            self.assertTrue(a.compose(b).equiv(Tensor(expected)))

    def test_tensor(self):
        """Test tensoring."""
        a = Tensor(2 * self.tensor)
        b = Tensor(3 * self.tensor)

        azeros = np.zeros((2, 2))
        azeros[0, 0] = 1
        amat = np.kron(azeros, 2 * self.expected)

        bzeros = np.zeros((2, 2))
        bzeros[1, 1] = 1
        bmat = np.kron(bzeros, 3 * self.expected)

        if isinstance(self.expected, np.ndarray):
            expected = np.einsum("ab,ij", amat, bmat)
        else:
            expected = np.einsum(
                "ab,ij", amat.todense(), bmat.todense()  # pylint: disable=no-member
            )

        self.assertTrue(a.tensor(b).equiv(Tensor(expected)))

    def test_expand(self):
        """Test expanding."""
        a = Tensor(2 * self.tensor)
        b = Tensor(3 * self.tensor)

        azeros = np.zeros((2, 2))
        azeros[0, 0] = 1
        amat = np.kron(azeros, 3 * self.expected)

        bzeros = np.zeros((2, 2))
        bzeros[1, 1] = 1
        bmat = np.kron(bzeros, 2 * self.expected)

        if isinstance(self.expected, np.ndarray):
            expected = np.einsum("ab,ij", amat, bmat)
        else:
            expected = np.einsum(
                "ab,ij", amat.todense(), bmat.todense()  # pylint: disable=no-member
            )

        self.assertTrue(a.tensor(b).equiv(Tensor(expected)))

    def test_coord_iter(self):
        """Test iterating."""
        for value, index in self.tensor.coord_iter():
            self.assertEqual(value, self.expected[index])

    def test_copy(self):
        """Test copying."""
        c = copy(self.tensor)
        self.assertEqual(self.tensor, c)
        self.assertNotEqual(id(self.tensor), id(c))
        self.assertEqual(id(self.tensor.array), id(c.array))

    def test_deepcopy(self):
        """Test deep-copying."""
        c = deepcopy(self.tensor)
        self.assertEqual(self.tensor, c)
        self.assertNotEqual(id(self.tensor), id(c))
        self.assertNotEqual(id(self.tensor.array), id(c.array))
