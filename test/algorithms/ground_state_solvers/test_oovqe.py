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

from typing import cast

from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.basicaer import BasicAer
from qiskit.utils import QuantumInstance

from qiskit_nature.algorithms.ground_state_solvers import OrbitalOptimizationVQE
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import \
    VQEUCCFactory
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.circuit.library.ansatzes import PUCCD
from qiskit_nature.drivers import HDF5Driver, QMolecule
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.molecular import MolecularProblem
from qiskit_nature.transformers import FreezeCoreTransformer


class TestOOVQE(QiskitNatureTestCase):
    """ Test OOVQE Ground State Calculation. """

    def setUp(self):
        super().setUp()

        self.driver1 = HDF5Driver(
            hdf5_input=self.get_resource_path('test_oovqe_h4.hdf5',
                                              'algorithms/ground_state_solvers'))
        self.driver2 = HDF5Driver(
            hdf5_input=self.get_resource_path('test_oovqe_lih.hdf5',
                                              'algorithms/ground_state_solvers'))
        self.driver3 = HDF5Driver(
            hdf5_input=self.get_resource_path('test_oovqe_h4_uhf.hdf5',
                                              'algorithms/ground_state_solvers'))

        self.qubit_converter = QubitConverter(JordanWignerMapper())

        self.molecular_problem1 = MolecularProblem(self.driver1)
        self.molecular_problem2 = MolecularProblem(self.driver2, [FreezeCoreTransformer()])
        self.molecular_problem3 = MolecularProblem(self.driver3)

        self.molecular_problem1.second_q_ops()
        self.molecular_problem2.second_q_ops()
        self.molecular_problem3.second_q_ops()

        self.energy1_rotation = -3.0104
        self.energy1 = -2.77  # energy of the VQE with pUCCD ansatz and LBFGSB optimizer
        self.energy2 = -7.70
        self.energy3 = -2.50
        self.initial_point1 = [0.039374, -0.47225463, -0.61891996, 0.02598386, 0.79045546,
                               -0.04134567, 0.04944946, -0.02971617, -0.00374005, 0.77542149]

        self.seed = 50
        self.optimizer = COBYLA(maxiter=1)
        self.quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                shots=1,
                                                seed_simulator=self.seed,
                                                seed_transpiler=self.seed)

    def test_orbital_rotations(self):
        """ Test that orbital rotations are performed correctly. """

        optimizer = COBYLA(maxiter=1)
        solver = VQEUCCFactory(
                    optimizer=optimizer,
                    quantum_instance=self.quantum_instance,
                    var_form=PUCCD(),
                    )
        oovqe = OrbitalOptimizationVQE(self.molecular_problem1, self.qubit_converter, solver,
                                       iterative_oo=False, initial_point=self.initial_point1)
        result = oovqe.solve(self.molecular_problem1)

        self.assertAlmostEqual(result.eigenenergies[0], self.energy1_rotation, 4)

    def test_oovqe(self):
        """ Test the simultaneous optimization of orbitals and ansatz parameters with OOVQE using
        BasicAer's statevector_simulator. """

        optimizer = COBYLA(maxiter=3, rhobeg=0.01)
        solver = VQEUCCFactory(
                    optimizer=optimizer,
                    quantum_instance=self.quantum_instance,
                    var_form=PUCCD(),
                    )
        oovqe = OrbitalOptimizationVQE(self.molecular_problem1, self.qubit_converter, solver,
                                       iterative_oo=False, initial_point=self.initial_point1)
        result = oovqe.solve(self.molecular_problem1)

        self.assertLessEqual(result.eigenenergies[0], self.energy1, 4)

    def test_iterative_oovqe(self):
        """ Test the iterative OOVQE using BasicAer's statevector_simulator. """

        optimizer = COBYLA(maxiter=2, rhobeg=0.01)
        solver = VQEUCCFactory(
                    optimizer=optimizer,
                    quantum_instance=self.quantum_instance,
                    var_form=PUCCD(),
                    )
        oovqe = OrbitalOptimizationVQE(self.molecular_problem1, self.qubit_converter, solver,
                                       iterative_oo=True, iterative_oo_iterations=2,
                                       initial_point=self.initial_point1)
        result = oovqe.solve(self.molecular_problem1)

        self.assertLessEqual(result.eigenenergies[0], self.energy1)

    @unittest.skip("Frozen-core support is currently broken.")
    def test_oovqe_with_frozen_core(self):
        """ Test the OOVQE with frozen core approximation. """

        optimizer = COBYLA(maxiter=2, rhobeg=1)

        solver = VQEUCCFactory(
                    optimizer=optimizer,
                    quantum_instance=self.quantum_instance,
                    var_form=PUCCD(),
                    )
        oovqe = OrbitalOptimizationVQE(self.molecular_problem2, self.qubit_converter, solver,
                                       iterative_oo=False)
        result = oovqe.solve(self.molecular_problem2)
        q_molecule_transformed = cast(QMolecule, self.molecular_problem2.molecule_data_transformed)

        self.assertLessEqual(result.eigenenergies[0] +
                             q_molecule_transformed.nuclear_repulsion_energy, self.energy2)

    def test_oovqe_with_unrestricted_hf(self):
        """ Test the OOVQE with unrestricted HF method. """

        optimizer = COBYLA(maxiter=2, rhobeg=0.01)
        solver = VQEUCCFactory(
                    optimizer=optimizer,
                    quantum_instance=self.quantum_instance,
                    var_form=PUCCD(),
                    )
        oovqe = OrbitalOptimizationVQE(self.molecular_problem3, self.qubit_converter, solver,
                                       iterative_oo=False, initial_point=self.initial_point1)
        result = oovqe.solve(self.molecular_problem3)

        self.assertLessEqual(result.eigenenergies, self.energy3)

    def test_oovqe_with_unsupported_varform(self):
        """ Test the OOVQE with unsupported varform. """

        q_molecule_transformed = cast(QMolecule, self.molecular_problem1.molecule_data_transformed)
        num_spin_orbitals = 2 * q_molecule_transformed.num_molecular_orbitals
        optimizer = COBYLA(maxiter=2, rhobeg=0.01)
        var_form = RealAmplitudes(num_qubits=num_spin_orbitals)
        solver = VQE(var_form=var_form, optimizer=optimizer,
                     quantum_instance=self.quantum_instance)
        oovqe = OrbitalOptimizationVQE(self.molecular_problem3, self.qubit_converter, solver,
                                       iterative_oo=False)
        with self.assertRaises(QiskitNatureError):
            oovqe.solve(self.molecular_problem1)

    def test_oovqe_with_vqe_uccsd(self):
        """ Test the OOVQE with VQE + UCCSD instead of factory. """

        optimizer = COBYLA(maxiter=3, rhobeg=0.01)
        q_molecule_transformed = cast(QMolecule, self.molecular_problem1.molecule_data_transformed)
        num_spin_orbitals = 2 * q_molecule_transformed.num_molecular_orbitals
        num_particles = (q_molecule_transformed.num_alpha, q_molecule_transformed.num_beta)
        initial_state = HartreeFock(num_spin_orbitals, num_particles, self.qubit_converter)
        var_form = PUCCD(qubit_converter=self.qubit_converter,
                         num_particles=num_particles,
                         num_spin_orbitals=num_spin_orbitals,
                         initial_state=initial_state)
        solver = VQE(var_form=var_form,
                     optimizer=optimizer,
                     quantum_instance=self.quantum_instance,
                     )
        oovqe = OrbitalOptimizationVQE(self.molecular_problem1,
                                       self.qubit_converter,
                                       solver,
                                       iterative_oo=False,
                                       initial_point=self.initial_point1
                                       )
        result = oovqe.solve(self.molecular_problem1)

        self.assertLessEqual(result.eigenenergies, self.energy1, 4)


if __name__ == '__main__':
    unittest.main()
