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
from test.random import random_quadratic_hamiltonian

import numpy as np
from ddt import data, ddt, unpack
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector, random_hermitian, random_unitary

from qiskit_nature.circuit.library import BogoliubovTransform
from qiskit_nature.second_quantization.operators import QubitConverter
from qiskit_nature.second_quantization.operators.fermionic import (
    BravyiKitaevMapper,
    JordanWignerMapper,
)
from qiskit_nature.second_quantization.operator_factories.quadratic_hamiltonian import (
    QuadraticHamiltonian,
)


def _expand_transformation_matrix(mat: np.ndarray) -> np.ndarray:
    n, _ = mat.shape
    left = mat[:, :n]
    right = mat[:, n:]
    return np.block([[left, right], [right.conj(), left.conj()]])


@ddt
class TestBogoliubovTransform(QiskitNatureTestCase):
    """Tests for BogoliubovTransform."""

    @unpack
    @data((4, True), (5, True), (4, False), (5, False))
    def test_bogoliubov_transform(self, n_orbitals, num_conserving):
        """Test Bogoliubov transform."""
        converter = QubitConverter(JordanWignerMapper())
        hamiltonian = random_quadratic_hamiltonian(
            n_orbitals, num_conserving=num_conserving, seed=5740
        )
        (
            transformation_matrix,
            orbital_energies,
            transformed_constant,
        ) = hamiltonian.diagonalizing_bogoliubov_transform()
        matrix = converter.map(hamiltonian.to_fermionic_op()).to_matrix()
        bog_circuit = BogoliubovTransform(transformation_matrix, qubit_converter=converter)
        for initial_state in range(2**n_orbitals):
            state = Statevector.from_int(initial_state, dims=2**n_orbitals)
            final_state = np.array(state.evolve(bog_circuit))
            occupied_orbitals = [i for i in range(n_orbitals) if initial_state >> i & 1]
            eig = np.sum(orbital_energies[occupied_orbitals]) + transformed_constant
            np.testing.assert_allclose(matrix @ final_state, eig * final_state, atol=1e-8)

    @data(4, 5)
    def test_bogoliubov_transform_compose_num_conserving(self, n_orbitals):
        """Test Bogoliubov transform composition, particle-number-conserving."""
        unitary1 = np.array(random_unitary(n_orbitals, seed=4331))
        unitary2 = np.array(random_unitary(n_orbitals, seed=2506))

        bog_circuit_1 = BogoliubovTransform(unitary1)
        bog_circuit_2 = BogoliubovTransform(unitary2)
        bog_circuit_composed = BogoliubovTransform(unitary1 @ unitary2)

        register = QuantumRegister(n_orbitals)
        circuit = QuantumCircuit(register)
        circuit.append(bog_circuit_1, register)
        circuit.append(bog_circuit_2, register)

        self.assertTrue(Operator(circuit).equiv(Operator(bog_circuit_composed), atol=1e-8))

    @data(4, 5)
    def test_bogoliubov_transform_compose_general(self, n_orbitals):
        """Test Bogoliubov transform composition, general."""
        hamiltonian1 = random_quadratic_hamiltonian(n_orbitals, num_conserving=False, seed=6990)
        hamiltonian2 = random_quadratic_hamiltonian(n_orbitals, num_conserving=False, seed=1447)
        transformation_matrix1, _, _ = hamiltonian1.diagonalizing_bogoliubov_transform()
        transformation_matrix2, _, _ = hamiltonian2.diagonalizing_bogoliubov_transform()
        composed_transformation_matrix = (
            _expand_transformation_matrix(transformation_matrix1)
            @ _expand_transformation_matrix(transformation_matrix2)
        )[:n_orbitals]

        bog_circuit_1 = BogoliubovTransform(transformation_matrix1)
        bog_circuit_2 = BogoliubovTransform(transformation_matrix2)
        bog_circuit_composed = BogoliubovTransform(composed_transformation_matrix)

        register = QuantumRegister(n_orbitals)
        circuit = QuantumCircuit(register)
        circuit.append(bog_circuit_1, register)
        circuit.append(bog_circuit_2, register)

        self.assertTrue(Operator(circuit).equiv(Operator(bog_circuit_composed), atol=1e-8))

    def test_no_side_effects(self):
        """Test that the routines don't mutate the input array."""
        n_orbitals = 5
        hermitian_part = random_hermitian(n_orbitals).data
        constant = np.random.uniform(-10, 10)

        quad_ham = QuadraticHamiltonian(hermitian_part, constant=constant)
        transformation_matrix, _, _ = quad_ham.diagonalizing_bogoliubov_transform()
        original = transformation_matrix.copy()
        _ = BogoliubovTransform(transformation_matrix)
        np.testing.assert_allclose(transformation_matrix, original, atol=1e-8)

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
