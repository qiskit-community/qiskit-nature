# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Test Nature logging methods
"""
from typing import List, Dict, Set
import sys
import unittest
import logging
import tempfile
import warnings
from test import QiskitNatureTestCase
from qiskit import BasicAer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit.library import TwoLocal
from qiskit_nature import settings as nature_settings
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
import qiskit_nature.optionals as _optionals


class TestHandler(logging.StreamHandler):
    """Unit Test Handler"""

    def __init__(self):
        """
        Initialize the handler.
        """
        super().__init__(sys.stdout)
        self.records: List[logging.LogRecord] = []

    def emit(self, record) -> None:
        """handle record"""
        self.records.append(record)


class TestLogging(QiskitNatureTestCase):
    """Test logging"""

    def setUp(self):
        super().setUp()
        self._test_handler = TestHandler()
        nature_settings.dict_aux_operators = True
        self._logging_dict = {"qiskit_nature": logging.DEBUG, "qiskit": logging.DEBUG}
        self._old_logging_dict = nature_settings.logging.get_levels_for_names(
            self._logging_dict.keys()
        )

        nature_settings.logging.set_levels_for_names(self._logging_dict, add_default_handler=False)
        for name in self._logging_dict:
            nature_settings.logging.add_handler(name, handler=self._test_handler)

    def tearDown(self) -> None:
        super().tearDown()
        for name in self._logging_dict:
            nature_settings.logging.remove_handler(name, handler=self._test_handler)
        nature_settings.logging.set_levels_for_names(
            self._old_logging_dict, add_default_handler=False
        )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_logging_to_handler(self):
        """logging test"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            TestLogging._vqe_run()

        # check that logging was handled
        name_levels: Dict[str, Set[int]] = {}
        for record in self._test_handler.records:
            names = record.name.split(".")
            if names[0] in name_levels:
                name_levels[names[0]].add(record.levelno)
            else:
                name_levels[names[0]] = set()

        self.assertCountEqual(
            name_levels.keys(),
            self._logging_dict.keys(),
            msg="Handled logging modules different from reference",
        )
        for name, ref_level in self._logging_dict.items():
            for level in name_levels[name]:
                self.assertGreaterEqual(
                    level,
                    ref_level,
                    msg=f"{name}: logging level {level} < reference level {ref_level} ",
                )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_logging_to_file(self):
        """logging test to file"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            nature_settings.logging.log_to_file(self._logging_dict.keys(), path=tmp_file.name)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                TestLogging._vqe_run()

            with open(tmp_file.name, encoding="utf8") as file:
                lines = file.read()

        for name in self._logging_dict:
            self.assertTrue(f"{name}." in lines, msg=f"name {name} not found in log file.")

    @staticmethod
    def _vqe_run():

        # Use PySCF, a classical computational chemistry software
        # package, to compute the one-body and two-body integrals in
        # electronic-orbital basis, necessary to form the Fermionic operator
        driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735", unit=UnitsType.ANGSTROM, basis="sto3g"
        )
        problem = ElectronicStructureProblem(driver)

        # generate the second-quantized operators
        second_q_ops = problem.second_q_ops()
        main_op = second_q_ops["ElectronicEnergy"]

        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")

        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals

        # setup the classical optimizer for VQE

        optimizer = L_BFGS_B()

        # setup the mapper and qubit converter

        mapper = ParityMapper()
        converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

        # map to qubit operators
        qubit_op = converter.convert(main_op, num_particles=num_particles)

        # setup the initial state for the ansatz

        init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

        # setup the ansatz for VQE

        ansatz = TwoLocal(num_spin_orbitals, ["ry", "rz"], "cz")

        # add the initial state
        ansatz.compose(init_state, front=True, inplace=True)

        # set the backend for the quantum computation

        backend = BasicAer.get_backend("statevector_simulator")

        # setup and run VQE

        algorithm = VQE(ansatz, optimizer=optimizer, quantum_instance=backend)

        result = algorithm.compute_minimum_eigenvalue(qubit_op)
        _ = problem.interpret(result)


if __name__ == "__main__":
    unittest.main()
