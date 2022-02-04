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

"""
Code inside the test is the chemistry sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest

from test import QiskitNatureTestCase
from qiskit.utils import optionals
import qiskit_nature.optionals as _optionals


class TestReadmeSample(QiskitNatureTestCase):
    """Test sample code from readme"""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_readme_sample(self):
        """readme sample test"""
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """overloads print to log values"""
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        from qiskit_nature.drivers import UnitsType
        from qiskit_nature.drivers.second_quantization import PySCFDriver
        from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

        # Use PySCF, a classical computational chemistry software
        # package, to compute the one-body and two-body integrals in
        # electronic-orbital basis, necessary to form the Fermionic operator
        driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735", unit=UnitsType.ANGSTROM, basis="sto3g"
        )
        problem = ElectronicStructureProblem(driver)

        # generate the second-quantized operators
        second_q_ops = problem.second_q_ops()
        main_op = second_q_ops[0]

        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")

        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals

        # setup the classical optimizer for VQE
        from qiskit.algorithms.optimizers import L_BFGS_B

        optimizer = L_BFGS_B()

        # setup the mapper and qubit converter
        from qiskit_nature.mappers.second_quantization import ParityMapper
        from qiskit_nature.converters.second_quantization import QubitConverter

        mapper = ParityMapper()
        converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

        # map to qubit operators
        qubit_op = converter.convert(main_op, num_particles=num_particles)

        # setup the initial state for the ansatz
        from qiskit_nature.circuit.library import HartreeFock

        init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

        # setup the ansatz for VQE
        from qiskit.circuit.library import TwoLocal

        ansatz = TwoLocal(num_spin_orbitals, ["ry", "rz"], "cz")

        # add the initial state
        ansatz.compose(init_state, front=True, inplace=True)

        # set the backend for the quantum computation
        from qiskit import Aer

        backend = Aer.get_backend("aer_simulator_statevector")

        # setup and run VQE
        from qiskit.algorithms import VQE

        algorithm = VQE(ansatz, optimizer=optimizer, quantum_instance=backend)

        result = algorithm.compute_minimum_eigenvalue(qubit_op)
        print(result.eigenvalue.real)

        electronic_structure_result = problem.interpret(result)
        print(electronic_structure_result)

        # ----------------------------------------------------------------------

        self.assertAlmostEqual(result.eigenvalue.real, -1.8572750301938803, places=6)


if __name__ == "__main__":
    unittest.main()
