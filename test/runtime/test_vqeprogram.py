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

"""Test the VQE program."""

from test import QiskitNatureTestCase

import unittest
import warnings
from ddt import ddt, data
import numpy as np
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.algorithms import VQEResult
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import I, Z

from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.second_q.operators import QubitConverter
from qiskit_nature.second_q.operators.fermionic import JordanWignerMapper
from qiskit_nature.runtime import VQEClient, VQEProgram, VQERuntimeResult, VQEProgramResult

from .fake_vqeruntime import FakeRuntimeProvider


@ddt
class TestVQEClient(QiskitNatureTestCase):
    """Test the VQE program."""

    def get_standard_program(self, use_deprecated=False):
        """Get a standard VQEClient and operator to find the ground state of."""
        circuit = RealAmplitudes(3)
        operator = Z ^ I ^ Z
        initial_point = np.random.random(circuit.num_parameters)
        backend = QasmSimulatorPy()

        if use_deprecated:
            vqe_cls = VQEProgram
            provider = FakeRuntimeProvider(use_deprecated=True)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        else:
            provider = FakeRuntimeProvider(use_deprecated=False)
            vqe_cls = VQEClient

        vqe = vqe_cls(
            ansatz=circuit,
            optimizer=SPSA(),
            initial_point=initial_point,
            backend=backend,
            provider=provider,
        )

        if use_deprecated:
            warnings.filterwarnings("always", category=DeprecationWarning)

        return vqe, operator

    @data({"name": "SPSA", "maxiter": 100}, SPSA(maxiter=100))
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        for use_deprecated in [False, True]:
            vqe, operator = self.get_standard_program(use_deprecated=use_deprecated)
            vqe.optimizer = optimizer
            result = vqe.compute_minimum_eigenvalue(operator)

            self.assertIsInstance(result, VQEResult)
            self.assertIsInstance(result, VQEProgramResult if use_deprecated else VQERuntimeResult)

    def test_supports_aux_ops(self):
        """Test the VQEClient says it supports aux operators."""
        for use_deprecated in [False, True]:
            vqe, _ = self.get_standard_program(use_deprecated=use_deprecated)
            self.assertTrue(vqe.supports_aux_operators)

    def test_return_groundstate(self):
        """Test the VQEClient yields a ground state solver that returns the ground state."""
        for use_deprecated in [False, True]:
            vqe, _ = self.get_standard_program(use_deprecated=use_deprecated)
            qubit_converter = QubitConverter(JordanWignerMapper())
            gss = GroundStateEigensolver(qubit_converter, vqe)
            self.assertTrue(gss.returns_groundstate)


if __name__ == "__main__":
    unittest.main()
