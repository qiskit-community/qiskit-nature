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

"""Test Bogoliubov transform circuits."""

from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit.quantum_info import Statevector, random_hermitian
from qiskit_nature.circuit.library import BogoliubovTransform
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.operators.second_quantization.quadratic_hamiltonian import QuadraticHamiltonian
from qiskit_nature.utils import random_antisymmetric_matrix


@ddt
class TestBogoliubovTransform(QiskitNatureTestCase):
    """Tests for BogoliubovTransform."""

    @data(4, 5)
    def test_bogoliubov_transform_num_conserving(self, n_orbitals):
        """Test particle-number-conserving Bogoliubov transform."""
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
        bog_circuit = BogoliubovTransform(transformation_matrix, qubit_converter=converter)
        for initial_state in range(2**n_orbitals):
            state = Statevector.from_int(initial_state, dims=2**n_orbitals)
            final_state = np.array(state.evolve(bog_circuit))
            occupied_orbitals = [i for i in range(n_orbitals) if initial_state >> i & 1]
            eig = np.sum(orbital_energies[occupied_orbitals]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-7)

    @data(4, 5)
    def test_bogoliubov_transform_general(self, n_orbitals):
        """Test general (non-particle-number-conserving) Bogoliubov transform."""
        converter = QubitConverter(JordanWignerMapper())
        hermitian_part = np.array(random_hermitian(n_orbitals))
        antisymmetric_part = random_antisymmetric_matrix(n_orbitals, seed=4744)
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
        bog_circuit = BogoliubovTransform(transformation_matrix, qubit_converter=converter)
        for initial_state in range(2**n_orbitals):
            state = Statevector.from_int(initial_state, dims=2**n_orbitals)
            final_state = np.array(state.evolve(bog_circuit))
            occupied_orbitals = [i for i in range(n_orbitals) if initial_state >> i & 1]
            eig = np.sum(orbital_energies[occupied_orbitals]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-7)

    def test_no_side_effects(self):
        """Test that the routines don't mutate the input array."""
        n_orbitals = 5
        hermitian_part = random_hermitian(n_orbitals).data
        constant = np.random.uniform(-10, 10)

        quad_ham = QuadraticHamiltonian(hermitian_part, constant=constant)
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        original = transformation_matrix.copy()
        _ = BogoliubovTransform(transformation_matrix)
        np.testing.assert_allclose(transformation_matrix, original, atol=1e-7)

    def test_validation(self):
        """Test input validation."""
        with self.assertRaisesRegex(ValueError, "2-dimensional"):
            _ = BogoliubovTransform(np.ones((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            _ = BogoliubovTransform(np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            _ = BogoliubovTransform(np.ones((2, 3)))

    def test_circuit_kwargs(self):
        """Test that circuit keyword arguments are actually passed through."""
        circuit = BogoliubovTransform(np.eye(2), name="abcd")
        assert circuit.name == "abcd"

    def test_unsupported_mapper(self):
        """Test passing unsupported mapper fails gracefully."""
        with self.assertRaisesRegex(NotImplementedError, "supported"):
            _ = BogoliubovTransform(np.eye(2), qubit_converter=QubitConverter(BravyiKitaevMapper()))
