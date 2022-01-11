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

""" Test of the Adaptive VQE ground state calculations """
import contextlib
import copy
import io
import unittest

from typing import cast

from test import QiskitNatureTestCase, requires_extra_library

import numpy as np

from qiskit.providers.basicaer import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.opflow.gradients import Gradient, NaturalGradient
from qiskit.test import slow_test

from qiskit_nature.algorithms import AdaptVQE, VQEUCCFactory
from qiskit_nature.circuit.library import HartreeFock, UCC
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ParticleNumber,
)
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)


class TestAdaptVQE(QiskitNatureTestCase):
    """Test Adaptive VQE Ground State Calculation"""

    @requires_extra_library
    def setUp(self):
        super().setUp()

        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735", unit=UnitsType.ANGSTROM, basis="sto3g"
        )

        self.problem = ElectronicStructureProblem(self.driver)

        self.expected = -1.85727503

        self.qubit_converter = QubitConverter(ParityMapper())

    def test_default(self):
        """Default execution"""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = AdaptVQE(self.qubit_converter, solver)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_param_shift(self):
        """test for param_shift method"""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        grad = Gradient(grad_method="param_shift")
        calc = AdaptVQE(self.qubit_converter, solver, gradient=grad)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_fin_diff(self):
        """test for fin_diff method"""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        grad = Gradient(grad_method="fin_diff", epsilon=1.0)
        calc = AdaptVQE(self.qubit_converter, solver, gradient=grad)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_lin_comb(self):
        """test for fin_diff method"""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        grad = Gradient(grad_method="lin_comb", epsilon=1.0)
        calc = AdaptVQE(self.qubit_converter, solver, gradient=grad)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_natural_gradients(self):
        """test for fin_diff method"""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        grad = NaturalGradient(
            grad_method="fin_diff", qfi_method="lin_comb_full", regularization="ridge"
        )
        calc = AdaptVQE(self.qubit_converter, solver, gradient=grad)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    @slow_test
    def test_LiH(self):
        """Lih test"""
        inter_dist = 1.6
        driver1 = PySCFDriver(
            atom="Li .0 .0 .0; H .0 .0 " + str(inter_dist),
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )
        transformer = ActiveSpaceTransformer(
            num_electrons=2,
            num_molecular_orbitals=3,
        )
        problem1 = ElectronicStructureProblem(driver1, [transformer])
        self.expected = -1.85727503
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = AdaptVQE(self.qubit_converter, solver)
        res = calc.solve(problem1)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_print_result(self):
        """Regression test against issues with printing results."""
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = AdaptVQE(self.qubit_converter, solver)
        res = calc.solve(self.problem)
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
              * Electronic dipole moment (a.u.): [0.0  0.0  1.38
                - computed part:      [0.0  0.0  1.38
              > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.
                             (debye): [0.0  0.0  0.0]  Total: 0.
        """
        for truth, expected in zip(out.getvalue().split("\n"), expected.split("\n")):
            assert truth.strip().startswith(expected.strip())

    def test_aux_ops_reusability(self):
        """Test that the auxiliary operators can be reused"""
        # Regression test against #1475
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = AdaptVQE(self.qubit_converter, solver)

        modes = 4
        h_1 = np.eye(modes, dtype=complex)
        h_2 = np.zeros((modes, modes, modes, modes))
        aux_ops = ElectronicEnergy(
            [
                OneBodyElectronicIntegrals(ElectronicBasis.MO, (h_1, None)),
                TwoBodyElectronicIntegrals(ElectronicBasis.MO, (h_2, None, None, None)),
            ]
        ).second_q_ops()
        aux_ops_copy = copy.deepcopy(aux_ops)

        _ = calc.solve(self.problem)
        assert all(
            frozenset(a.to_list()) == frozenset(b.to_list()) for a, b in zip(aux_ops, aux_ops_copy)
        )

    def test_custom_minimum_eigensolver(self):
        """Test custom MES"""

        class CustomFactory(VQEUCCFactory):
            """A custom MESFactory"""

            def get_solver(self, problem, qubit_converter):
                particle_number = cast(
                    ParticleNumber,
                    problem.grouped_property_transformed.get_property(ParticleNumber),
                )
                num_spin_orbitals = particle_number.num_spin_orbitals
                num_particles = (particle_number.num_alpha, particle_number.num_beta)

                initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)
                ansatz = UCC(
                    qubit_converter=qubit_converter,
                    num_particles=num_particles,
                    num_spin_orbitals=num_spin_orbitals,
                    excitations="d",
                    initial_state=initial_state,
                )
                vqe = VQE(
                    ansatz=ansatz,
                    quantum_instance=self.quantum_instance,
                    optimizer=L_BFGS_B(),
                )
                return vqe

        solver = CustomFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))

        calc = AdaptVQE(self.qubit_converter, solver)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_custom_excitation_pool(self):
        """Test custom excitation pool"""

        class CustomFactory(VQEUCCFactory):
            """A custom MES factory."""

            def get_solver(self, problem, qubit_converter):
                solver = super().get_solver(problem, qubit_converter)
                # Here, we can create essentially any custom excitation pool.
                # For testing purposes only, we simply select some hopping operator already
                # available in the ansatz object.
                # pylint: disable=no-member
                custom_excitation_pool = [solver.ansatz.operators[2]]
                solver.ansatz.operators = custom_excitation_pool
                return solver

        solver = CustomFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = AdaptVQE(self.qubit_converter, solver)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)

    def test_vqe_adapt_check_cyclicity(self):
        """AdaptVQE index cycle detection"""
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


if __name__ == "__main__":
    unittest.main()
