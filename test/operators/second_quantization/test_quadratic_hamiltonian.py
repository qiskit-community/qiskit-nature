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

"""Tests for QuadraticHamiltonian"""

from test import QiskitNatureTestCase
from test.random import random_antisymmetric_matrix

import numpy as np
from ddt import data, ddt
from qiskit.quantum_info import random_hermitian

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import QuadraticHamiltonian
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp


@ddt
class TestQuadraticHamiltonian(QiskitNatureTestCase):
    """QuadraticHamiltonian tests."""

    def test_init(self):
        """Test initialization."""
        hermitian_part = np.eye(2)
        antisymmetric_part = np.array([[0, 1], [-1, 0]])
        constant = 1.0
        zero = np.zeros((2, 2))

        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant)
        np.testing.assert_allclose(quad_ham.hermitian_part, hermitian_part)
        np.testing.assert_allclose(quad_ham.antisymmetric_part, antisymmetric_part)
        np.testing.assert_allclose(quad_ham.constant, constant)
        self.assertEqual(quad_ham.num_modes, 2)

        quad_ham = QuadraticHamiltonian(None, antisymmetric_part)
        np.testing.assert_allclose(quad_ham.hermitian_part, zero)
        self.assertEqual(quad_ham.num_modes, 2)

        quad_ham = QuadraticHamiltonian(hermitian_part)
        np.testing.assert_allclose(quad_ham.antisymmetric_part, zero)
        self.assertEqual(quad_ham.num_modes, 2)

        quad_ham = QuadraticHamiltonian(num_modes=2)
        np.testing.assert_allclose(quad_ham.hermitian_part, zero)
        np.testing.assert_allclose(quad_ham.antisymmetric_part, zero)
        self.assertEqual(quad_ham.num_modes, 2)

        with self.assertRaisesRegex(ValueError, "specified"):
            _ = QuadraticHamiltonian()

    def test_conserves_particle_number(self):
        """Test particle number conservation predicate."""
        hermitian_part = np.eye(2)
        antisymmetric_part = np.array([[0, 1], [-1, 0]])

        quad_ham = QuadraticHamiltonian(hermitian_part)
        assert quad_ham.conserves_particle_number()

        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part)
        assert not quad_ham.conserves_particle_number()

    def test_diagonalizing_bogoliubov_transform(self):
        """Test diagonalizing Bogoliubov transform."""
        hermitian_part = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=complex
        )
        antisymmetric_part = np.array(
            [[0.0, 1.0j, 0.0], [-1.0j, 0.0, 1.0j], [0.0, -1.0j, 0.0]], dtype=complex
        )
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part)
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = quad_ham.diagonalizing_bogoliubov_transform()

        # test that the transformation diagonalizes the Hamiltonian
        left = transformation_matrix[:, :3]
        right = transformation_matrix[:, 3:]
        full_transformation_matrix = np.block([[left, right], [right.conj(), left.conj()]])
        eye = np.eye(3, dtype=complex)
        majorana_basis = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
        basis_change = majorana_basis @ full_transformation_matrix @ majorana_basis.T.conj()
        majorana_matrix, majorana_constant = quad_ham.majorana_form()
        canonical = basis_change @ majorana_matrix @ basis_change.T

        zero = np.zeros((3, 3))
        diagonal = np.diag(orbital_energies)
        expected = np.block([[zero, diagonal], [-diagonal, zero]])

        np.testing.assert_allclose(orbital_energies, np.sort(orbital_energies))
        np.testing.assert_allclose(canonical, expected, atol=1e-7)
        np.testing.assert_allclose(
            transformed_constant, majorana_constant - 0.5 * np.sum(orbital_energies)
        )

        # confirm eigenvalues match with Jordan-Wigner transformed Hamiltonian
        hamiltonian_jw = (
            QubitConverter(mapper=JordanWignerMapper())
            .convert(quad_ham.to_fermionic_op())
            .primitive.to_matrix()
        )
        eigs, _ = np.linalg.eigh(hamiltonian_jw)
        expected_eigs = np.array(
            [
                np.sum(orbital_energies[list(occupied_orbitals)]) + transformed_constant
                for occupied_orbitals in [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
            ]
        )
        np.testing.assert_allclose(np.sort(eigs), np.sort(expected_eigs), atol=1e-7)

    @data(1, 2, 3, 4, 5, 6)
    def test_diagonalizing_bogoliubov_transform_particle_number_conserving(self, n_orbitals):
        """Test diagonalizing Bogoliubov transform, particle-number-conserving case."""
        hermitian_part = random_hermitian(n_orbitals).data
        constant = np.random.uniform(-10, 10)
        quad_ham = QuadraticHamiltonian(hermitian_part, constant=constant)
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = quad_ham.diagonalizing_bogoliubov_transform()
        diagonalized = transformation_matrix @ hermitian_part.T @ transformation_matrix.T.conj()

        np.testing.assert_allclose(orbital_energies, np.sort(orbital_energies))
        np.testing.assert_allclose(diagonalized, np.diag(orbital_energies), atol=1e-7)
        np.testing.assert_allclose(transformed_constant, constant)

    @data(2, 3, 4, 5, 6)
    def test_diagonalizing_bogoliubov_transform_non_particle_number_conserving(self, n_orbitals):
        """Test diagonalizing Bogoliubov transform, non-particle-number-conserving case."""
        hermitian_part = random_hermitian(n_orbitals).data
        antisymmetric_part = random_antisymmetric_matrix(n_orbitals, seed=9985)
        constant = np.random.uniform(-10, 10)
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant=constant)
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = quad_ham.diagonalizing_bogoliubov_transform()

        left = transformation_matrix[:, :n_orbitals]
        right = transformation_matrix[:, n_orbitals:]
        full_transformation_matrix = np.block([[left, right], [right.conj(), left.conj()]])
        eye = np.eye(n_orbitals, dtype=complex)
        majorana_basis = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
        basis_change = majorana_basis @ full_transformation_matrix @ majorana_basis.T.conj()
        majorana_matrix, majorana_constant = quad_ham.majorana_form()
        canonical = basis_change @ majorana_matrix @ basis_change.T

        zero = np.zeros((n_orbitals, n_orbitals))
        diagonal = np.diag(orbital_energies)
        expected = np.block([[zero, diagonal], [-diagonal, zero]])

        np.testing.assert_allclose(orbital_energies, np.sort(orbital_energies))
        np.testing.assert_allclose(canonical, expected, atol=1e-7)
        np.testing.assert_allclose(
            transformed_constant, majorana_constant - 0.5 * np.sum(orbital_energies)
        )

    def test_fermionic_op(self):
        """Test conversion to FermionicOp."""
        hermitian_part = np.array([[1, 2j], [-2j, 3]])
        antisymmetric_part = np.array([[0, 4j], [-4j, 0]])
        constant = 5.0
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant)
        fermionic_op = quad_ham.to_fermionic_op()
        expected_terms = [
            ("NI", 1.0),
            ("IN", 3.0),
            ("+-", 2j),
            ("-+", 2j),
            ("++", 4j),
            ("--", 4j),
            ("II", 5.0),
        ]
        expected_op = FermionicOp(expected_terms)
        matrix = fermionic_op.to_matrix(sparse=False)
        expected_matrix = expected_op.to_matrix(sparse=False)
        np.testing.assert_allclose(matrix, expected_matrix)

    def test_validate(self):
        """Test input validation."""
        mat = np.array([[1, 2], [3, 4]])
        _ = QuadraticHamiltonian(hermitian_part=mat, antisymmetric_part=None, validate=False)
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            _ = QuadraticHamiltonian(hermitian_part=mat, antisymmetric_part=None)
        with self.assertRaisesRegex(ValueError, "Antisymmetric"):
            _ = QuadraticHamiltonian(hermitian_part=None, antisymmetric_part=mat)

        hermitian_part = np.array([[1, 2j], [-2j, 3]])
        antisymmetric_part = np.array([[0, 4, 0], [-4, 0, 4], [0, -4, 0]])
        with self.assertRaisesRegex(ValueError, "same shape"):
            _ = QuadraticHamiltonian(hermitian_part, antisymmetric_part)
        with self.assertRaisesRegex(ValueError, "num_modes"):
            _ = QuadraticHamiltonian(hermitian_part, num_modes=5)
        with self.assertRaisesRegex(ValueError, "num_modes"):
            _ = QuadraticHamiltonian(antisymmetric_part=antisymmetric_part, num_modes=5)
