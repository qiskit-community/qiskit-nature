# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
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

from qiskit import BasicAer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import ExcitationPreserving
from qiskit.test import slow_test
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem


@slow_test
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

    def test_excitation_preserving(self):
        """Test the excitation preserving wavefunction on a chemistry example."""

        driver = HDF5Driver(
            self.get_resource_path("test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d")
        )

        converter = QubitConverter(ParityMapper())

        problem = ElectronicStructureProblem(driver)

        _ = problem.second_q_ops()

        num_particles = (
            problem.driver_result_transformed.get_property("ParticleNumber").num_alpha,
            problem.driver_result_transformed.get_property("ParticleNumber").num_beta,
        )

        num_spin_orbitals = problem.molecule_data_transformed.num_molecular_orbitals * 2

        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(num_spin_orbitals, num_particles, converter)

        wavefunction = ExcitationPreserving(num_spin_orbitals)
        wavefunction.compose(initial_state, front=True, inplace=True)

        solver = VQE(
            ansatz=wavefunction,
            optimizer=optimizer,
            quantum_instance=QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )

        gsc = GroundStateEigensolver(converter, solver)

        result = gsc.solve(problem)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=4)


if __name__ == "__main__":
    unittest.main()
