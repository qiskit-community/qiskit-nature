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

"""Test for SparseLabelOp"""

import unittest
import warnings
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack

from qiskit_nature.operators.second_quantization import SparseLabelOp

# create dummy subclass

class TestSparseLabelOp(QiskitNatureTestCase):
    """SparseLabelOp tests."""

    def test_add(self):
        """Test add method"""

    def test_mul(self):
        """Test scalar multiplication method"""

    def test_adjoint(self):
        """Test adjoint method"""
    
    def test_compose(self):
        """Test compose method"""
    
    def test_commutativity(self):
        """Test commutativity method"""
    
    def test_normal_order(self):
        """test normal_order method"""
