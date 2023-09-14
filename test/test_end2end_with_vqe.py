# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test End to End with VQE """

import unittest

from test import QiskitNatureTestCase

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
class TestEnd2End(QiskitNatureTestCase):
    """End2End VQE tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 42

        driver = PySCFDriver()
        problem = driver.run()
        main_op, aux_ops = problem.second_q_ops()
        mapper = ParityMapper(num_particles=problem.num_particles)
        self.qubit_op = mapper.map(main_op)
        self.aux_ops = mapper.map(aux_ops)
        self.reference_energy = -1.857275027031588

    def test_end2end_h2(self):
        """end to end h2"""
        optimizer = COBYLA(maxiter=1000)
        ryrz = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        vqe = VQE(Estimator(), ryrz, optimizer)
        result = vqe.compute_minimum_eigenvalue(self.qubit_op, aux_operators=self.aux_ops)
        self.assertAlmostEqual(result.eigenvalue.real, self.reference_energy, places=4)


if __name__ == "__main__":
    unittest.main()
