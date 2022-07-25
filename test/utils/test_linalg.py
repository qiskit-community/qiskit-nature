# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test linear algebra utilities."""

from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack
from qiskit_nature.utils import givens_matrix


@ddt
class TestGivensMatrix(QiskitNatureTestCase):
    """Tests for computing Givens rotation matrix."""

    @unpack
    @data((0, 1 + 1j), (1 + 1j, 0), (1 + 2j, 3 - 4j))
    def test_givens_matrix(self, a: complex, b: complex):
        """Test computing Givens rotation matrix."""
        givens_mat = givens_matrix(a, b)
        product = givens_mat @ np.array([a, b])
        np.testing.assert_allclose(product[1], 0.0, atol=1e-8)
