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

"""Test Slater determinant state preparation circuits."""

from test import QiskitNatureTestCase
from test.random import random_quadratic_hamiltonian

import numpy as np
from qiskit.quantum_info import Statevector

from qiskit_nature.second_q.circuit.library import SlaterDeterminant
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
)


class TestSlaterDeterminant(QiskitNatureTestCase):
    """Tests for preparing Slater determinants."""

    def test_slater_determinant(self):
        """Test preparing Slater determinants."""
        n_orbitals = 5
        converter = QubitConverter(JordanWignerMapper())
        quad_ham = random_quadratic_hamiltonian(n_orbitals, num_conserving=True, seed=8839)
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = quad_ham.diagonalizing_bogoliubov_transform()
        fermionic_op = quad_ham.to_fermionic_op()
        qubit_op = converter.convert(fermionic_op)
        matrix = qubit_op.to_matrix()
        for n_particles in range(n_orbitals + 1):
            circuit = SlaterDeterminant(
                transformation_matrix[:n_particles], qubit_converter=converter
            )
            final_state = np.array(Statevector(circuit))
            eig = np.sum(orbital_energies[:n_particles]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-7)

    def test_no_side_effects(self):
        """Test that the routines don't mutate the input array."""
        n_orbitals = 5
        n_particles = 3
        quad_ham = random_quadratic_hamiltonian(n_orbitals, num_conserving=True, seed=8839)
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        original = transformation_matrix.copy()
        _ = SlaterDeterminant(transformation_matrix[:n_particles])
        np.testing.assert_allclose(transformation_matrix, original, atol=1e-7)

    def test_validation(self):
        """Test input validation."""
        with self.assertRaisesRegex(ValueError, "2-dimensional"):
            _ = SlaterDeterminant(np.ones((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            _ = SlaterDeterminant(np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            _ = SlaterDeterminant(np.ones((2, 3)))

    def test_circuit_kwargs(self):
        """Test that circuit keyword arguments are actually passed through."""
        circuit = SlaterDeterminant(np.eye(2), name="abcd")
        assert circuit.name == "abcd"

    def test_unsupported_mapper(self):
        """Test passing unsupported mapper fails gracefully."""
        with self.assertRaisesRegex(NotImplementedError, "supported"):
            _ = SlaterDeterminant(np.eye(2), qubit_converter=QubitConverter(BravyiKitaevMapper()))
