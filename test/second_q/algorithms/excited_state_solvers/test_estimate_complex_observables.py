# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for _estimate_complex_observables."""

import unittest
from test import QiskitNatureTestCase

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_nature.second_q.algorithms.excited_states_solvers.qeom import (
    _estimate_complex_observables,
)


class TestEstimateComplexObservables(QiskitNatureTestCase):
    """Tests for the _estimate_complex_observables helper in qeom.py.

    All tests use the |0> state and Z/Y Pauli operators whose expectation values
    can be computed analytically: <0|Z|0> = 1, <0|Y|0> = 0, <0|X|0> = 0.
    """

    def setUp(self):
        super().setUp()
        self.estimator = StatevectorEstimator()
        self.circuit = QuantumCircuit(1)  # |0> state, no gates

    def test_real_operator(self):
        """Purely real coefficient passes through and gives a real expectation value."""
        result = _estimate_complex_observables(
            self.estimator, self.circuit, {"z": SparsePauliOp("Z")}
        )
        self.assertAlmostEqual(result["z"][0].real, 1.0)
        self.assertAlmostEqual(result["z"][0].imag, 0.0)

    def test_imaginary_operator(self):
        """Purely imaginary coefficient."""
        # O = i*Z  =>  <0|O|0> = i*<0|Z|0> = i
        result = _estimate_complex_observables(
            self.estimator, self.circuit, {"iz": SparsePauliOp(["Z"], [1j])}
        )
        self.assertAlmostEqual(result["iz"][0].real, 0.0)
        self.assertAlmostEqual(result["iz"][0].imag, 1.0)

    def test_mixed_complex_operator(self):
        """Mixed (real + imaginary) coefficients give a complex expectation value."""
        # O = (1+i)*Z  =>  <0|O|0> = 1+i
        result = _estimate_complex_observables(
            self.estimator, self.circuit, {"mixed": SparsePauliOp(["Z"], [1 + 1j])}
        )
        self.assertAlmostEqual(result["mixed"][0].real, 1.0)
        self.assertAlmostEqual(result["mixed"][0].imag, 1.0)

    def test_none_values_excluded_from_results(self):
        """None operators are skipped and absent from the returned dict."""
        obs = {"z": SparsePauliOp("Z"), "skip": None}
        result = _estimate_complex_observables(self.estimator, self.circuit, obs)
        self.assertIn("z", result)
        self.assertNotIn("skip", result)


if __name__ == "__main__":
    unittest.main()
