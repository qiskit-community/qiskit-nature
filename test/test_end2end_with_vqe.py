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

""" Test End to End with VQE """

import unittest

from test import QiskitNatureTestCase

from qiskit import BasicAer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem


class TestEnd2End(QiskitNatureTestCase):
    """End2End VQE tests."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 42

        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        problem = ElectronicStructureProblem(driver)
        second_q_ops = problem.second_q_ops()
        converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        num_particles = (
            problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
            problem.grouped_property_transformed.get_property("ParticleNumber").num_beta,
        )
        main_op = second_q_ops[0]
        self.qubit_op = converter.convert(main_op, num_particles)
        self.aux_ops = converter.convert_match(second_q_ops)
        self.reference_energy = -1.857275027031588

    def test_end2end_h2(self):
        """end to end h2"""
        backend = BasicAer.get_backend("statevector_simulator")
        shots = 1
        optimizer = COBYLA(maxiter=1000)
        ryrz = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        quantum_instance = QuantumInstance(backend, shots=shots)
        vqe = VQE(ryrz, optimizer=optimizer, quantum_instance=quantum_instance)
        result = vqe.compute_minimum_eigenvalue(self.qubit_op, aux_operators=self.aux_ops)
        self.assertAlmostEqual(result.eigenvalue.real, self.reference_energy, places=4)


if __name__ == "__main__":
    unittest.main()
