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

"""Test random sampling utilities."""

import itertools
from test import QiskitNatureTestCase
from test.random import random_two_body_tensor

import numpy as np

from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    TwoBodyElectronicIntegrals,
)


class TestRandomTwoBodyTensor(QiskitNatureTestCase):
    """Test random two-body tensor."""

    def test_random_two_body_tensor_hermitian(self):
        """Test random two-body tensor is hermitian."""
        n_orbitals = 5
        two_body_tensor = random_two_body_tensor(n_orbitals)
        two_body_integrals = TwoBodyElectronicIntegrals(ElectronicBasis.SO, two_body_tensor)
        # TODO maybe need to permute integrals from physicist to chemist
        op = two_body_integrals.to_second_q_op()
        self.assertTrue(op.is_hermitian())

    def test_random_two_body_tensor_positive(self):
        """Test random two-body tensor is hermitian."""
        n_orbitals = 5
        two_body_tensor = random_two_body_tensor(n_orbitals)
        reshaped_tensor = two_body_tensor.reshape((n_orbitals**2, n_orbitals**2))
        np.testing.assert_allclose(reshaped_tensor, reshaped_tensor.T)
        eigs, _ = np.linalg.eigh(reshaped_tensor)
        np.testing.assert_array_less(-1e-8, eigs)

    def test_random_two_body_tensor_symmetry(self):
        """Test random two-body tensor symmetry."""
        n_orbitals = 5
        two_body_tensor = random_two_body_tensor(n_orbitals)
        for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
            val = two_body_tensor[p, q, r, s]
            self.assertAlmostEqual(two_body_tensor[r, s, p, q], val)
            self.assertAlmostEqual(two_body_tensor[q, p, s, r], val.conjugate())
            self.assertAlmostEqual(two_body_tensor[s, r, q, p], val.conjugate())
            self.assertAlmostEqual(two_body_tensor[q, p, r, s], val)
            self.assertAlmostEqual(two_body_tensor[s, r, p, q], val)
            self.assertAlmostEqual(two_body_tensor[p, q, s, r], val)
            self.assertAlmostEqual(two_body_tensor[r, s, q, p], val)
