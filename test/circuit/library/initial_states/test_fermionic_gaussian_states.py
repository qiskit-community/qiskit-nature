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

"""Test fermionic Gaussian state preparation circuits."""

from test import QiskitNatureTestCase

import numpy as np
from qiskit.quantum_info import random_hermitian, Statevector
from qiskit_nature.circuit.library import FermionicGaussianState, SlaterDeterminant
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization.quadratic_hamiltonian import QuadraticHamiltonian


def _random_antisymmetric(dim: int):
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return mat - mat.T


class TestSlaterDeterminant(QiskitNatureTestCase):
    """Tests for preparing Slater determinants."""

    def test_slater_determinant(self):
        """Test preparing Slater determinants."""
        n_orbitals = 5
        converter = QubitConverter(JordanWignerMapper())
        hermitian_part = random_hermitian(n_orbitals).data
        constant = np.random.uniform(-10, 10)
        quad_ham = QuadraticHamiltonian(hermitian_part, constant=constant)
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = quad_ham.diagonalizing_bogoliubov_transform()
        fermionic_op = quad_ham.to_fermionic_op()
        qubit_op = converter.convert(fermionic_op)
        matrix = qubit_op.to_matrix()
        for n_particles in range(n_orbitals + 1):
            circuit = SlaterDeterminant(transformation_matrix[:n_particles], converter)
            final_state = np.array(Statevector(circuit))
            eig = np.sum(orbital_energies[:n_particles]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-7)

    def test_fermionic_gaussian_state(self):
        """Test preparing fermionic Gaussian states."""
        n_orbitals = 5
        converter = QubitConverter(JordanWignerMapper())
        hermitian_part = random_hermitian(n_orbitals).data
        antisymmetric_part = _random_antisymmetric(n_orbitals)
        constant = np.random.uniform(-10, 10)
        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant)
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = quad_ham.diagonalizing_bogoliubov_transform()
        fermionic_op = quad_ham.to_fermionic_op()
        qubit_op = converter.convert(fermionic_op)
        matrix = qubit_op.to_matrix()
        occupied_orbitals_lists = [
            [],
            [0],
            [3],
            [0, 1],
            [2, 4],
            [1, 3, 4],
            range(n_orbitals),
        ]
        for occupied_orbitals in occupied_orbitals_lists:
            circuit = FermionicGaussianState(transformation_matrix, occupied_orbitals, converter)
            final_state = np.array(Statevector(circuit))
            eig = np.sum(orbital_energies[occupied_orbitals]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-7)

    def test_no_side_effects(self):
        """Test that the routines don't mutate the input array."""
        n_orbitals = 5
        n_particles = 3
        converter = QubitConverter(JordanWignerMapper())
        hermitian_part = random_hermitian(n_orbitals).data
        antisymmetric_part = _random_antisymmetric(n_orbitals)
        constant = np.random.uniform(-10, 10)

        quad_ham = QuadraticHamiltonian(hermitian_part, constant=constant)
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        original = transformation_matrix.copy()
        _ = SlaterDeterminant(transformation_matrix[:n_particles], qubit_converter=converter)
        np.testing.assert_allclose(transformation_matrix, original, atol=1e-7)

        quad_ham = QuadraticHamiltonian(hermitian_part, antisymmetric_part, constant=constant)
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        original = transformation_matrix.copy()
        _ = FermionicGaussianState(
            transformation_matrix, occupied_orbitals=[2, 3], qubit_converter=converter
        )
        np.testing.assert_allclose(transformation_matrix, original, atol=1e-7)

    def test_circuit_kwargs(self):
        """Test that circuit keyword arguments are actually passed through."""
        circuit = SlaterDeterminant(np.eye(2), name="abcd")
        assert circuit.name == "abcd"

        circuit = FermionicGaussianState(
            np.array([[0.5, 0.5, 0.5, -0.5], [0.5, 0.5, -0.5, 0.5]]), name="efgh"
        )
        assert circuit.name == "efgh"
