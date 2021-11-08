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

""" Test GroundStateEigensolver """

import contextlib
import copy
import io
import unittest
import warnings

from test import QiskitNatureTestCase

import numpy as np

from qiskit import BasicAer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.opflow import AerPauliExpectation, PauliExpectation
from qiskit.test import slow_test
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit_nature import settings
from qiskit_nature.algorithms import (
    GroundStateEigensolver,
    VQEUCCFactory,
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.circuit.library import HartreeFock, UCC, UCCSD
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer


class TestGroundStateEigensolver(QiskitNatureTestCase):
    """Test GroundStateEigensolver"""

    def setUp(self):
        super().setUp()
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*drivers.*")
        self.driver = HDF5Driver(
            self.get_resource_path("test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d")
        )
        self.seed = 56
        algorithm_globals.random_seed = self.seed

        self.reference_energy = -1.1373060356951838

        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = ElectronicStructureProblem(self.driver)

        self.num_spin_orbitals = 4
        self.num_particles = (1, 1)

    def test_npme(self):
        """Test NumPyMinimumEigensolver"""
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_npme_with_default_filter(self):
        """Test NumPyMinimumEigensolver with default filter"""
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd(self):
        """Test VQE UCCSD case"""
        solver = VQEUCCFactory(
            quantum_instance=QuantumInstance(BasicAer.get_backend("statevector_simulator")),
            ansatz=UCC(excitations="d"),
        )
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd_with_callback(self):
        """Test VQE UCCSD with callback."""

        def callback(nfev, parameters, energy, stddev):
            # pylint: disable=unused-argument
            print(f"iterations {nfev}: energy: {energy}")

        solver = VQEUCCFactory(
            quantum_instance=QuantumInstance(BasicAer.get_backend("statevector_simulator")),
            callback=callback,
        )
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)
        for idx, line in enumerate(out.getvalue().split("\n")):
            if line.strip():
                self.assertTrue(line.startswith(f"iterations {idx+1}: energy: "))

    def test_vqe_ucc_custom(self):
        """Test custom ansatz in Factory use case"""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_aux_ops_reusability(self):
        """Test that the auxiliary operators can be reused"""
        # Regression test against #1475
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)

        modes = 4
        h_1 = np.eye(modes, dtype=complex)
        h_2 = np.zeros((modes, modes, modes, modes))
        aux_ops = ElectronicEnergy(
            [
                OneBodyElectronicIntegrals(ElectronicBasis.MO, (h_1, None)),
                TwoBodyElectronicIntegrals(ElectronicBasis.MO, (h_2, None, None, None)),
            ],
        ).second_q_ops()
        aux_ops_copy = copy.deepcopy(aux_ops)

        _ = calc.solve(self.electronic_structure_problem)
        assert all(
            frozenset(a.to_list()) == frozenset(b.to_list()) for a, b in zip(aux_ops, aux_ops_copy)
        )

    def test_dict_based_aux_ops(self):
        """Test the `test_dict_based_aux_ops` variant"""
        try:
            settings.dict_aux_operators = True
            solver = NumPyMinimumEigensolverFactory()
            calc = GroundStateEigensolver(self.qubit_converter, solver)
            res = calc.solve(self.electronic_structure_problem)
            self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)
            self.assertTrue(
                np.all(isinstance(aux_op, dict) for aux_op in res.aux_operator_eigenvalues)
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["ParticleNumber"][0], 2.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["ParticleNumber"][1], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["AngularMomentum"][0], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["AngularMomentum"][1], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["Magnetization"][0], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["Magnetization"][1], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["DipoleMomentX"][0], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["DipoleMomentX"][1], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["DipoleMomentY"][0], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["DipoleMomentY"][1], 0.0, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["DipoleMomentZ"][0], -1.3889487, places=6
            )
            self.assertAlmostEqual(
                res.aux_operator_eigenvalues[0]["DipoleMomentZ"][1], 0.0, places=6
            )
        finally:
            settings.dict_aux_operators = False

    def _setup_evaluation_operators(self):
        # first we run a ground state calculation
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)

        # now we decide that we want to evaluate another operator
        # for testing simplicity, we just use some pre-constructed auxiliary operators
        _, *aux_ops = self.qubit_converter.convert_match(
            self.electronic_structure_problem.second_q_ops()
        )
        return calc, res, aux_ops

    def test_eval_op_single(self):
        """Test evaluating a single additional operator"""
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because in this test we test a single operator evaluation
        add_aux_op = aux_ops[0][0]

        # now we have the ground state calculation evaluate it
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res[0], complex)
        self.assertAlmostEqual(add_aux_op_res[0].real, 2, places=6)

    def test_eval_op_single_none(self):
        """Test evaluating a single `None` operator"""
        calc, res, _ = self._setup_evaluation_operators()
        # we filter the list because in this test we test a single operator evaluation
        add_aux_op = None

        # now we have the ground state calculation evaluate it
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsNone(add_aux_op_res)

    def test_eval_op_list(self):
        """Test evaluating a list of additional operators"""
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because of simplicity
        expected_results = {"number of particles": 2, "s^2": 0, "magnetization": 0}
        add_aux_op = aux_ops[0:3]

        # now we have the ground state calculation evaluate them
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res, list)
        # in this list we require that the order of the results remains unchanged
        for idx, expected in enumerate(expected_results.values()):
            self.assertAlmostEqual(add_aux_op_res[idx][0].real, expected, places=6)

    def test_eval_op_list_none(self):
        """Test evaluating a list of additional operators incl. `None`"""
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because of simplicity
        expected_results = {"number of particles": 2, "s^2": 0, "magnetization": 0}
        add_aux_op = aux_ops[0:3] + [None]

        # now we have the ground state calculation evaluate them
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res, list)
        # in this list we require that the order of the results remains unchanged
        for idx, expected in enumerate(expected_results.values()):
            self.assertAlmostEqual(add_aux_op_res[idx][0].real, expected, places=6)
        self.assertIsNone(add_aux_op_res[-1])

    def test_eval_op_dict(self):
        """Test evaluating a dict of additional operators"""
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because of simplicity
        expected_results = {"number of particles": 2, "s^2": 0, "magnetization": 0}
        add_aux_op = aux_ops[0:3]
        # now we convert it into a dictionary
        add_aux_op = dict(zip(expected_results.keys(), add_aux_op))

        # now we have the ground state calculation evaluate them
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res, dict)
        for name, expected in expected_results.items():
            self.assertAlmostEqual(add_aux_op_res[name][0].real, expected, places=6)

    def test_eval_op_dict_none(self):
        """Test evaluating a dict of additional operators incl. `None`"""
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because of simplicity
        expected_results = {"number of particles": 2, "s^2": 0, "magnetization": 0}
        add_aux_op = aux_ops[0:3]
        # now we convert it into a dictionary
        add_aux_op = dict(zip(expected_results.keys(), add_aux_op))
        add_aux_op["None"] = None

        # now we have the ground state calculation evaluate them
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res, dict)
        for name, expected in expected_results.items():
            self.assertAlmostEqual(add_aux_op_res[name][0].real, expected, places=6)
        self.assertIsNone(add_aux_op_res["None"])

    @slow_test
    def test_eval_op_qasm(self):
        """Regression tests against https://github.com/Qiskit/qiskit-nature/issues/53."""
        solver = VQEUCCFactory(
            optimizer=SLSQP(maxiter=100),
            expectation=PauliExpectation(),
            quantum_instance=QuantumInstance(
                backend=BasicAer.get_backend("qasm_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res_qasm = calc.solve(self.electronic_structure_problem)

        hamiltonian = self.electronic_structure_problem.second_q_ops()[0]
        qubit_op = self.qubit_converter.map(hamiltonian)

        ansatz = solver.get_solver(self.electronic_structure_problem, self.qubit_converter).ansatz
        circuit = ansatz.assign_parameters(res_qasm.raw_result.optimal_point)
        mean = calc.evaluate_operators(circuit, qubit_op)

        self.assertAlmostEqual(res_qasm.eigenenergies[0], mean[0].real)

    def test_eval_op_qasm_aer(self):
        """Regression tests against https://github.com/Qiskit/qiskit-nature/issues/53."""
        try:
            # pylint: disable=import-outside-toplevel
            # pylint: disable=unused-import
            from qiskit import Aer

            backend = Aer.get_backend("aer_simulator")
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

        solver = VQEUCCFactory(
            optimizer=SLSQP(maxiter=100),
            expectation=AerPauliExpectation(),
            include_custom=True,
            quantum_instance=QuantumInstance(
                backend=backend,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res_qasm = calc.solve(self.electronic_structure_problem)

        hamiltonian = self.electronic_structure_problem.second_q_ops()[0]
        qubit_op = self.qubit_converter.map(hamiltonian)

        ansatz = solver.get_solver(self.electronic_structure_problem, self.qubit_converter).ansatz
        circuit = ansatz.assign_parameters(res_qasm.raw_result.optimal_point)
        mean = calc.evaluate_operators(circuit, qubit_op)

        self.assertAlmostEqual(res_qasm.eigenenergies[0], mean[0].real)

    def _prepare_uccsd_hf(self, qubit_converter):
        initial_state = HartreeFock(self.num_spin_orbitals, self.num_particles, qubit_converter)
        ansatz = UCCSD(
            qubit_converter,
            self.num_particles,
            self.num_spin_orbitals,
            initial_state=initial_state,
        )

        return ansatz

    def test_uccsd_hf(self):
        """uccsd hf test"""
        ansatz = self._prepare_uccsd_hf(self.qubit_converter)

        optimizer = SLSQP(maxiter=100)
        backend = BasicAer.get_backend("statevector_simulator")
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=QuantumInstance(backend=backend),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)

    @slow_test
    def test_uccsd_hf_qasm(self):
        """uccsd hf test with qasm simulator."""
        qubit_converter = QubitConverter(ParityMapper())
        ansatz = self._prepare_uccsd_hf(qubit_converter)

        backend = BasicAer.get_backend("qasm_simulator")

        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            expectation=PauliExpectation(),
            quantum_instance=QuantumInstance(
                backend=backend,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )

        gsc = GroundStateEigensolver(qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(result.total_energies[0], -1.138, places=2)

    @slow_test
    def test_uccsd_hf_aer_statevector(self):
        """uccsd hf test with Aer statevector"""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer

            backend = Aer.get_backend("aer_simulator_statevector")
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

        ansatz = self._prepare_uccsd_hf(self.qubit_converter)

        optimizer = SLSQP(maxiter=100)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=QuantumInstance(backend=backend),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)

    @slow_test
    def test_uccsd_hf_aer_qasm(self):
        """uccsd hf test with Aer qasm simulator."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer

            backend = Aer.get_backend("aer_simulator")
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

        ansatz = self._prepare_uccsd_hf(self.qubit_converter)

        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            expectation=PauliExpectation(group_paulis=False),
            quantum_instance=QuantumInstance(
                backend=backend,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(result.total_energies[0], -1.131, places=2)

    @slow_test
    def test_uccsd_hf_aer_qasm_snapshot(self):
        """uccsd hf test with Aer qasm simulator snapshot."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer

            backend = Aer.get_backend("aer_simulator")
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

        ansatz = self._prepare_uccsd_hf(self.qubit_converter)

        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            expectation=AerPauliExpectation(),
            quantum_instance=QuantumInstance(backend=backend),
        )

        gsc = GroundStateEigensolver(self.qubit_converter, solver)

        result = gsc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=3)

    def test_freeze_core_z2_symmetry_compatibility(self):
        """Regression test against #192.

        An issue arose when the FreezeCoreTransformer was combined with the automatic Z2Symmetry
        reduction. This regression test ensures that this behavior remains fixed.
        """
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        problem = ElectronicStructureProblem(driver, [FreezeCoreTransformer()])
        qubit_converter = QubitConverter(
            ParityMapper(),
            two_qubit_reduction=True,
            z2symmetry_reduction="auto",
        )

        solver = NumPyMinimumEigensolverFactory()
        gsc = GroundStateEigensolver(qubit_converter, solver)

        result = gsc.solve(problem)
        self.assertAlmostEqual(result.total_energies[0], -7.882, places=2)

    def test_total_dipole(self):
        """Regression test against #198.

        An issue with calculating the dipole moment that had division None/float.
        """
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        self.assertAlmostEqual(res.total_dipole_moment_in_debye[0], 0.0, places=1)

    def test_print_result(self):
        """Regression test against #198 and general issues with printing results."""
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.qubit_converter, solver)
        res = calc.solve(self.electronic_structure_problem)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            print(res)
        # do NOT change the below! Lines have been truncated as to not force exact numerical matches
        expected = """\
            === GROUND STATE ENERGY ===

            * Electronic ground state energy (Hartree): -1.857
              - computed part:      -1.857
            ~ Nuclear repulsion energy (Hartree): 0.719
            > Total ground state energy (Hartree): -1.137

            === MEASURED OBSERVABLES ===

              0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000

            === DIPOLE MOMENTS ===

            ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.38

              0:
              * Electronic dipole moment (a.u.): [0.0  0.0  -1.38
                - computed part:      [0.0  0.0  -1.38
              > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.
                             (debye): [0.0  0.0  0.0]  Total: 0.
        """
        for truth, expected in zip(out.getvalue().split("\n"), expected.split("\n")):
            assert truth.strip().startswith(expected.strip())


if __name__ == "__main__":
    unittest.main()
