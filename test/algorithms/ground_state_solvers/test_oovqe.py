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

""" Test of the OOVQE ground state calculations """
import unittest

from test import QiskitNatureTestCase

from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.basicaer import BasicAer
from qiskit.utils import QuantumInstance

from qiskit_nature.algorithms.ground_state_solvers.oovqe_algorithm import (
    OrbitalOptimizationVQE,
    CustomProblem,
)
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import (
    VQEUCCFactory,
)
from qiskit_nature.circuit.library.ansatzes import PUCCD
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter


class TestOOVQE(QiskitNatureTestCase):
    """Test OOVQE Ground State Calculation."""

    def setUp(self):
        super().setUp()

        self.driver1 = HDF5Driver(
            hdf5_input=self.get_resource_path("algorithms/ground_state_solvers/test_oovqe_h4.hdf5")
        )
        self.driver2 = HDF5Driver(
            hdf5_input=self.get_resource_path("algorithms/ground_state_solvers/test_oovqe_lih.hdf5")
        )
        self.driver3 = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "algorithms/ground_state_solvers/test_oovqe_h4_uhf.hdf5"
            )
        )

        self.qubit_converter = QubitConverter(JordanWignerMapper())

        self.molecular_problem1 = CustomProblem(self.driver1)
        self.molecular_problem2 = CustomProblem(self.driver2)
        self.molecular_problem3 = CustomProblem(self.driver3)

        self.molecular_problem1.second_q_ops()
        self.molecular_problem2.second_q_ops()
        self.molecular_problem3.second_q_ops()

        self.energy1_rotation = -3.0104
        self.energy1 = -2.77  # energy of the VQE with pUCCD ansatz and LBFGSB optimizer
        self.energy2 = -7.70
        self.energy3 = -2.50
        self.initial_point1 = [
            0.039374,
            -0.47225463,
            -0.61891996,
            0.02598386,
            0.79045546,
            -0.04134567,
            0.04944946,
            -0.02971617,
            -0.00374005,
            0.77542149,
        ]

        self.seed = 50
        self.optimizer1 = COBYLA(maxiter=1)
        self.quantum_instance = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

    def test_orbital_rotations(self):
        """Test that orbital rotations are performed correctly."""

        optimizer = COBYLA(maxiter=1)
        ansatz = PUCCD(self.qubit_converter, num_particles=[2, 2], num_spin_orbitals=8, reps=1)

        solver = VQEUCCFactory(
            optimizer=optimizer,
            quantum_instance=self.quantum_instance,
            ansatz=ansatz,
        )
        oovqe = OrbitalOptimizationVQE(
            qubit_converter=self.qubit_converter, solver=solver, initial_point=self.initial_point1
        )
        result = oovqe.solve(self.molecular_problem1)

        self.assertAlmostEqual(result.eigenenergies[0], self.energy1_rotation, 4)

    def test_oovqe(self):
        """Test the simultaneous optimization of orbitals and ansatz parameters with OOVQE using
        BasicAer's statevector_simulator."""

        optimizer = COBYLA(maxiter=3, rhobeg=0.01)
        ansatz = PUCCD(self.qubit_converter, num_particles=[2, 2], num_spin_orbitals=8, reps=1)

        solver = VQEUCCFactory(
            optimizer=optimizer,
            quantum_instance=self.quantum_instance,
            ansatz=ansatz,
        )

        oovqe = OrbitalOptimizationVQE(
            qubit_converter=self.qubit_converter, solver=solver, initial_point=self.initial_point1
        )

        result = oovqe.solve(self.molecular_problem1)

        self.assertLessEqual(result.eigenenergies[0], self.energy1, 4)


if __name__ == "__main__":
    unittest.main()
