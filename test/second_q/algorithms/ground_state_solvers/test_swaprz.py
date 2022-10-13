# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of ExcitationPreserving from the circuit library."""

import unittest
from test import QiskitNatureTestCase

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit.circuit.library import ExcitationPreserving
from qiskit.test import slow_test
from qiskit.utils import algorithm_globals
import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter


class TestExcitationPreserving(QiskitNatureTestCase):
    """The ExcitationPresering wavefunction was design to preserve the excitation of the system.

    We test it here from chemistry with JORDAN_WIGNER mapping (then the number of particles
    is preserved) and HartreeFock initial state to set it up. This facilitates testing
    ExcitationPreserving using these chemistry components/problem to ensure its correct operation.
    """

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.reference_energy = -1.137305593252385

    @slow_test
    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_excitation_preserving(self):
        """Test the excitation preserving wavefunction on a chemistry example."""

        driver = PySCFDriver()

        converter = QubitConverter(ParityMapper())

        problem = driver.run()

        _ = problem.second_q_ops()

        num_particles = problem.num_particles
        num_spatial_orbitals = problem.num_spatial_orbitals

        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(num_spatial_orbitals, num_particles, converter)

        num_qubits = 2 * num_spatial_orbitals
        wavefunction = ExcitationPreserving(int(num_qubits))
        wavefunction.compose(initial_state, front=True, inplace=True)

        solver = VQE(
            ansatz=wavefunction,
            optimizer=optimizer,
            estimator=Estimator(),
        )

        gsc = GroundStateEigensolver(converter, solver)

        result = gsc.solve(problem)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=4)


if __name__ == "__main__":
    unittest.main()
