# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of the Adaptive VQE implementation with the adaptive UCCSD variational form """

import unittest
from test import QiskitNatureTestCase

from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.opflow import TwoQubitReduction
from qiskit_nature import FermionicOperator
from qiskit_nature.algorithms.ground_state_solvers import AdaptVQE, VQEUCCSDFactory
from qiskit_nature.transformations import FermionicTransformation
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.components.variational_forms import UCCSD
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature import QiskitNatureError


@unittest.skip("Skip test until refactored.")
class TestAdaptVQEUCCSD(QiskitNatureTestCase):
    """ Test Adaptive VQE with UCCSD"""

    def setUp(self):
        super().setUp()
        # np.random.seed(50)
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                      unit=UnitsType.ANGSTROM,
                                      basis='sto3g')
        except QiskitNatureError:
            self.skipTest('PYSCF driver does not appear to be installed')
            return

        molecule = self.driver.run()
        self.num_particles = molecule.num_alpha + molecule.num_beta
        self.num_spin_orbitals = molecule.num_molecular_orbitals * 2
        fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
        map_type = 'PARITY'
        qubit_op = fer_op.mapping(map_type)
        self.qubit_op = TwoQubitReduction(num_particles=self.num_particles).convert(qubit_op)
        self.num_qubits = self.qubit_op.num_qubits
        converter = QubitConverter(mappers=ParityMapper())
        self.init_state = HartreeFock(self.num_spin_orbitals, self.num_particles, converter)
        self.var_form_base = None

    def test_uccsd_adapt(self):
        """ UCCSD test for adaptive features """
        self.var_form_base = UCCSD(self.num_spin_orbitals,
                                   self.num_particles, initial_state=self.init_state)
        self.var_form_base.manage_hopping_operators()
        # assert that the excitation pool exists
        self.assertIsNotNone(self.var_form_base.excitation_pool)
        # assert that the hopping ops list has been reset to be empty
        self.assertEqual(self.var_form_base._hopping_ops, [])

    def test_vqe_adapt(self):
        """ AdaptVQE test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
            backend = Aer.get_backend('statevector_simulator')
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        class CustomFactory(VQEUCCSDFactory):
            """A custom MESFactory"""

            def get_solver(self, transformation):
                num_orbitals = transformation.molecule_info['num_orbitals']
                num_particles = transformation.molecule_info['num_particles']
                converter = QubitConverter(mappers=ParityMapper())
                initial_state = HartreeFock(num_orbitals, num_particles, converter)
                var_form = UCCSD(num_orbitals,
                                 num_particles,
                                 initial_state=initial_state)
                vqe = VQE(var_form=var_form, quantum_instance=self._quantum_instance,
                          optimizer=L_BFGS_B())
                return vqe

        algorithm = AdaptVQE(FermionicTransformation(),
                             solver=CustomFactory(QuantumInstance(backend)),
                             threshold=0.00001,
                             delta=0.1,
                             max_iterations=1)
        result = algorithm.solve(driver=self.driver)
        self.assertEqual(result.num_iterations, 1)
        self.assertEqual(result.finishing_criterion, 'Maximum number of iterations reached')

        algorithm = AdaptVQE(FermionicTransformation(),
                             solver=CustomFactory(QuantumInstance(backend)),
                             threshold=0.00001,
                             delta=0.1)
        result = algorithm.solve(driver=self.driver)
        self.assertAlmostEqual(result.electronic_energies[0], -1.85727503, places=2)
        self.assertEqual(result.num_iterations, 2)
        self.assertAlmostEqual(result.final_max_gradient, 0.0, places=5)
        self.assertEqual(result.finishing_criterion, 'Threshold converged')

    def test_vqe_adapt_check_cyclicity(self):
        """ AdaptVQE index cycle detection """
        param_list = [
            ([1, 1], True),
            ([1, 11], False),
            ([11, 1], False),
            ([1, 12], False),
            ([12, 2], False),
            ([1, 1, 1], True),
            ([1, 2, 1], False),
            ([1, 2, 2], True),
            ([1, 2, 21], False),
            ([1, 12, 2], False),
            ([11, 1, 2], False),
            ([1, 2, 1, 1], True),
            ([1, 2, 1, 2], True),
            ([1, 2, 1, 21], False),
            ([11, 2, 1, 2], False),
            ([1, 11, 1, 111], False),
            ([11, 1, 111, 1], False),
            ([1, 2, 3, 1, 2, 3], True),
            ([1, 2, 3, 4, 1, 2, 3], False),
            ([11, 2, 3, 1, 2, 3], False),
            ([1, 2, 3, 1, 2, 31], False),
            ([1, 2, 3, 4, 1, 2, 3, 4], True),
            ([11, 2, 3, 4, 1, 2, 3, 4], False),
            ([1, 2, 3, 4, 1, 2, 3, 41], False),
            ([1, 2, 3, 4, 5, 1, 2, 3, 4], False),
        ]
        for seq, is_cycle in param_list:
            with self.subTest(msg="Checking index cyclicity in:", seq=seq):
                self.assertEqual(is_cycle, AdaptVQE._check_cyclicity(seq))


if __name__ == '__main__':
    unittest.main()
