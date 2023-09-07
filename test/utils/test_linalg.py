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

"""Test linear algebra utilities."""

import unittest
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack
from qiskit.quantum_info import random_unitary

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators.symmetric_two_body import unfold
from qiskit_nature.testing import random_two_body_tensor_real
from qiskit_nature.utils import double_factorized, givens_matrix, modified_cholesky


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


@ddt
class TestModifiedCholesky(QiskitNatureTestCase):
    """Tests for modified Cholesky decomposition."""

    @data(4, 5)
    def test_modified_cholesky(self, dim: int):
        """Test modified Cholesky decomposition on a random tensor."""
        rng = np.random.default_rng(4640)
        # construct a random positive definite matrix
        unitary = np.array(random_unitary(dim, seed=rng))
        eigs = rng.uniform(size=dim)
        mat = unitary @ np.diag(eigs) @ unitary.T.conj()
        cholesky_vecs = modified_cholesky(mat)
        reconstructed = np.einsum("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
        np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@ddt
class TestLowRankTwoBodyDecomposition(QiskitNatureTestCase):
    """Tests for low rank two-body decomposition."""

    @data(4, 5)
    def test_double_factorized_random(self, dim: int):
        """Test low rank two-body decomposition on a random tensor."""
        two_body_tensor = random_two_body_tensor_real(dim, seed=25257)
        diag_coulomb_mats, orbital_rotations = double_factorized(two_body_tensor)
        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            orbital_rotations,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_double_factorized_error_threshold_max_vecs(self):
        """Test low rank decomposition error threshold and max rank."""
        driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.6")
        driver_result = driver.run()
        electronic_energy = driver_result.hamiltonian
        two_body_tensor = unfold(electronic_energy.electronic_integrals.alpha["++--"])

        with self.subTest("max rank"):
            max_vecs = 20
            diag_coulomb_mats, orbital_rotations = double_factorized(
                two_body_tensor, max_vecs=max_vecs
            )
            reconstructed = np.einsum(
                "tpk,tqk,tkl,trl,tsl->pqrs",
                orbital_rotations,
                orbital_rotations,
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations,
            )
            self.assertEqual(len(orbital_rotations), max_vecs)
            np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-5)

        with self.subTest("error threshold"):
            error_threshold = 1e-4
            diag_coulomb_mats, orbital_rotations = double_factorized(
                two_body_tensor, error_threshold=error_threshold
            )
            reconstructed = np.einsum(
                "tpk,tqk,tkl,trl,tsl->pqrs",
                orbital_rotations,
                orbital_rotations,
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations,
            )
            self.assertLessEqual(len(orbital_rotations), 18)
            np.testing.assert_allclose(reconstructed, two_body_tensor, atol=error_threshold)

        with self.subTest("error threshold and max rank"):
            diag_coulomb_mats, orbital_rotations = double_factorized(
                two_body_tensor, error_threshold=error_threshold, max_vecs=max_vecs
            )
            reconstructed = np.einsum(
                "tpk,tqk,tkl,trl,tsl->pqrs",
                orbital_rotations,
                orbital_rotations,
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations,
            )
            self.assertLessEqual(len(orbital_rotations), 18)
            np.testing.assert_allclose(reconstructed, two_body_tensor, atol=error_threshold)
