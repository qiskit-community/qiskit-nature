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

"""Test fermionic Gaussian state preparation circuits."""

from test import QiskitNatureTestCase
from test.random import random_quadratic_hamiltonian

import numpy as np
from qiskit.quantum_info import Statevector

from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import BravyiKitaevMapper, JordanWignerMapper


class TestFermionicGaussianState(QiskitNatureTestCase):
    """Tests for preparing fermionic Gaussian states determinants."""

    def test_fermionic_gaussian_state(self):
        """Test preparing fermionic Gaussian states."""
        n_orbitals = 5
        converter = QubitConverter(JordanWignerMapper())
        quad_ham = random_quadratic_hamiltonian(n_orbitals, seed=5957)
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
            circuit = FermionicGaussianState(
                transformation_matrix, occupied_orbitals, qubit_converter=converter
            )
            final_state = np.array(Statevector(circuit))
            eig = np.sum(orbital_energies[occupied_orbitals]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-7)

    def test_no_side_effects(self):
        """Test that the routines don't mutate the input array."""
        n_orbitals = 5
        quad_ham = random_quadratic_hamiltonian(n_orbitals, seed=8353)
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        original = transformation_matrix.copy()
        _ = FermionicGaussianState(transformation_matrix, occupied_orbitals=[2, 3])
        np.testing.assert_allclose(transformation_matrix, original, atol=1e-7)

    def test_validation(self):
        """Test input validation."""
        with self.assertRaisesRegex(ValueError, "2-dimensional"):
            _ = FermionicGaussianState(np.ones((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            _ = FermionicGaussianState(np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "valid"):
            _ = FermionicGaussianState(np.ones((2, 4)))

    def test_circuit_kwargs(self):
        """Test that circuit keyword arguments are actually passed through."""
        circuit = FermionicGaussianState(
            np.array([[0.5, 0.5, 0.5, -0.5], [0.5, 0.5, -0.5, 0.5]]), name="efgh"
        )
        assert circuit.name == "efgh"

    def test_unsupported_mapper(self):
        """Test passing unsupported mapper fails gracefully."""
        with self.assertRaisesRegex(NotImplementedError, "supported"):
            _ = FermionicGaussianState(
                np.block([np.eye(2), np.zeros((2, 2))]),
                qubit_converter=QubitConverter(BravyiKitaevMapper()),
            )
