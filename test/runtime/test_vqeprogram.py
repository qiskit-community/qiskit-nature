# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the VQE program."""

from test import QiskitNatureDeprecatedTestCase
from test.runtime.fake_vqeruntime import FakeRuntimeProvider

import unittest
import warnings
from ddt import ddt, data
import numpy as np
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.providers.fake_provider import FakeArmonk, FakeArmonkV2
from qiskit.algorithms import VQEResult
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import I, Z

from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.runtime import VQEClient, VQERuntimeResult


@ddt
class TestVQEClient(QiskitNatureDeprecatedTestCase):
    """Test the VQE program."""

    def get_standard_program(self):
        """Get a standard VQEClient and operator to find the ground state of."""
        circuit = RealAmplitudes(3)
        operator = Z ^ I ^ Z
        initial_point = np.random.random(circuit.num_parameters)
        backend = QasmSimulatorPy()

        provider = FakeRuntimeProvider()
        vqe_cls = VQEClient

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            vqe = vqe_cls(
                ansatz=circuit,
                optimizer=SPSA(),
                initial_point=initial_point,
                backend=backend,
                provider=provider,
            )

        return vqe, operator

    @data({"name": "SPSA", "maxiter": 100}, SPSA(maxiter=100))
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        vqe, operator = self.get_standard_program()
        vqe.optimizer = optimizer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            result = vqe.compute_minimum_eigenvalue(operator)

        self.assertIsInstance(result, VQEResult)
        self.assertIsInstance(result, VQERuntimeResult)

    def test_supports_aux_ops(self):
        """Test the VQEClient says it supports aux operators."""
        vqe, _ = self.get_standard_program()
        self.assertTrue(vqe.supports_aux_operators)

    def test_return_groundstate(self):
        """Test the VQEClient yields a ground state solver that returns the ground state."""
        vqe, _ = self.get_standard_program()
        qubit_converter = QubitConverter(JordanWignerMapper())
        gss = GroundStateEigensolver(qubit_converter, vqe)
        self.assertTrue(gss.returns_groundstate)

    @data(FakeArmonk, FakeArmonkV2)
    def test_v1_and_v2_compatibility(self, backend_cls):
        """Test the VQE client is compatible both with V1 and V2 backends.

        Regression test of github.com/Qiskit/qiskit-nature/issues/1100.
        """
        provider = FakeRuntimeProvider()
        backend = backend_cls()

        circuit = RealAmplitudes(1, reps=0)
        operator = Z
        initial_point = np.array([0])
        optimizer = SPSA(maxiter=1, perturbation=0.1, learning_rate=0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            vqe = VQEClient(circuit, optimizer, initial_point, provider, backend)
            result = vqe.compute_minimum_eigenvalue(operator)

        self.assertIsInstance(result, VQERuntimeResult)


if __name__ == "__main__":
    unittest.main()
