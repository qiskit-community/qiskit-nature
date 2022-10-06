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

"""Tests for the EigenstateResult."""

import unittest
from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.second_q.problems import EigenstateResult


class TestEigenstateResult(QiskitNatureTestCase):
    """Tests EigenstateResult"""

    def test_groundenergy(self):
        """Tests ground energy"""
        eigenstate_result = EigenstateResult()
        eigenstate_result.eigenvalues = np.array([1, 2, 3])
        self.assertEqual(eigenstate_result.groundenergy, 1)


if __name__ == "__main__":
    unittest.main()
