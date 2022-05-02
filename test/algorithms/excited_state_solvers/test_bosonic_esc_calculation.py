# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Numerical qEOM excited states calculation """

import contextlib
import io
import unittest
import warnings

from test import QiskitNatureTestCase

import qiskit
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA

from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.drivers.second_quantization import VibrationalStructureDriver
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization.vibrational import (
    VibrationalStructureProblem,
)

from qiskit_nature.algorithms import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
    VQEUVCCFactory,
    QEOM,
    ExcitedStatesEigensolver,
    NumPyEigensolverFactory,
)
from qiskit_nature.properties.second_quantization.vibrational import (
    VibrationalStructureDriverResult,
)


class _DummyBosonicDriver(VibrationalStructureDriver):
    def __init__(self):
        super().__init__()
        modes = [
            [605.3643675, 1, 1],
            [-605.3643675, -1, -1],
            [340.5950575, 2, 2],
            [-340.5950575, -2, -2],
            [-89.09086530649508, 2, 1, 1],
            [-15.590557244410897, 2, 2, 2],
            [1.6512647916666667, 1, 1, 1, 1],
            [5.03965375, 2, 2, 1, 1],
            [0.43840625000000005, 2, 2, 2, 2],
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            watson = WatsonHamiltonian(modes, 2)
            self._driver_result = VibrationalStructureDriverResult.from_legacy_driver_result(watson)

    def run(self):
        """Run dummy driver to return test watson hamiltonian"""
        return self._driver_result


class TestBosonicESCCalculation(QiskitNatureTestCase):
    """Test Numerical QEOM excited states calculation"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.reference_energies = [
            1889.95738428,
            3294.21806197,
            4287.26821341,
            5819.76975784,
        ]

        self.driver = _DummyBosonicDriver()
        self.qubit_converter = QubitConverter(DirectMapper())
        self.basis_size = 2
        self.truncation_order = 2

        self.vibrational_problem = VibrationalStructureProblem(
            self.driver, self.basis_size, self.truncation_order
        )

    def test_numpy_mes(self):
        """Test with NumPyMinimumEigensolver"""
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, "sd")
        results = esc.solve(self.vibrational_problem)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_vibrational_energies[idx], energy, places=4)

    def test_numpy_factory(self):
        """Test with NumPyEigensolver"""
        solver = NumPyEigensolverFactory(use_default_filter_criterion=True)
        esc = ExcitedStatesEigensolver(self.qubit_converter, solver)
        results = esc.solve(self.vibrational_problem)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_vibrational_energies[idx], energy, places=4)

    def test_vqe_uvccsd_factory(self):
        """Test with VQE plus UVCCSD"""
        optimizer = COBYLA(maxiter=5000)
        quantum_instance = QuantumInstance(
            backend=qiskit.BasicAer.get_backend("statevector_simulator"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        solver = VQEUVCCFactory(quantum_instance, optimizer=optimizer)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, "sd")
        results = esc.solve(self.vibrational_problem)
        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_vibrational_energies[idx], energy, places=1)

    def test_vqe_uvccsd_with_callback(self):
        """Test VQE UVCCSD with callback."""

        def cb_callback(nfev, parameters, energy, stddev):
            print(f"iterations {nfev}: energy: {energy}")

        optimizer = COBYLA(maxiter=5000)

        quantum_instance = QuantumInstance(
            backend=qiskit.BasicAer.get_backend("statevector_simulator"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        solver = VQEUVCCFactory(quantum_instance, optimizer=optimizer, callback=cb_callback)
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, "sd")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            results = esc.solve(self.vibrational_problem)
        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_vibrational_energies[idx], energy, places=1)
        for idx, line in enumerate(out.getvalue().split("\n")):
            if line.strip():
                self.assertTrue(line.startswith(f"iterations {idx+1}: energy: "))


if __name__ == "__main__":
    unittest.main()
