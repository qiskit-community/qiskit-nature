# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Numerical qEOM excited states calculation """

import unittest

from test import QiskitNatureTestCase

from qiskit import BasicAer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA

from qiskit_nature.drivers.second_quantization import BosonicDriver, WatsonHamiltonian
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


class _DummyBosonicDriver(BosonicDriver):
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
        self._watson = WatsonHamiltonian(modes, 2)

    def run(self):
        """Run dummy driver to return test watson hamiltonian"""
        return self._watson


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

        for idx in enumerate(self.reference_energies):
            self.assertAlmostEqual(
                results.computed_vibrational_energies[idx],
                self.reference_energies[idx],
                places=4,
            )

    def test_numpy_factory(self):
        """Test with NumPyEigensolver"""
        solver = NumPyEigensolverFactory(use_default_filter_criterion=True)
        esc = ExcitedStatesEigensolver(self.qubit_converter, solver)
        results = esc.solve(self.vibrational_problem)

        for idx in enumerate(self.reference_energies):
            self.assertAlmostEqual(
                results.computed_vibrational_energies[idx],
                self.reference_energies[idx],
                places=4,
            )

    def test_vqe_uvccsd_factory(self):
        """Test with VQE plus UVCCSD"""
        optimizer = COBYLA(maxiter=5000)
        solver = VQEUVCCFactory(
            QuantumInstance(BasicAer.get_backend("statevector_simulator")),
            optimizer=optimizer,
        )
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, "sd")
        results = esc.solve(self.vibrational_problem)
        for idx in enumerate(self.reference_energies):
            self.assertAlmostEqual(
                results.computed_vibrational_energies[idx],
                self.reference_energies[idx],
                places=1,
            )


if __name__ == "__main__":
    unittest.main()
