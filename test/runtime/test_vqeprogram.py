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

"""Test the VQE program."""

from test import QiskitNatureTestCase

import unittest
from ddt import ddt, data
import numpy as np
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.algorithms import VQEResult
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import I, Z

from qiskit_nature.runtime import VQEProgram

from .fake_vqeruntime import FakeRuntimeProvider


@ddt
class TestVQEProgram(QiskitNatureTestCase):
    """Test the VQE program."""

    def setUp(self):
        super().setUp()
        self.provider = FakeRuntimeProvider()

    @data({"name": "SPSA", "maxiter": 100}, SPSA(maxiter=100))
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        circuit = RealAmplitudes(3)
        operator = Z ^ I ^ Z
        initial_point = np.random.random(circuit.num_parameters)
        backend = QasmSimulatorPy()

        vqe = VQEProgram(
            ansatz=circuit,
            optimizer=optimizer,
            initial_point=initial_point,
            backend=backend,
            provider=self.provider,
        )
        result = vqe.compute_minimum_eigenvalue(operator)

        self.assertIsInstance(result, VQEResult)


if __name__ == "__main__":
    unittest.main()
