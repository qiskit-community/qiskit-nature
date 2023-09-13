# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests Electronic Structure Problem."""
import unittest
from test import QiskitNatureTestCase

import json
import numpy as np

from qiskit_algorithms import MinimumEigensolverResult

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import AngularMomentum, Magnetization, ParticleNumber
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


class TestElectronicStructureProblem(QiskitNatureTestCase):
    """Tests Electronic Structure Problem."""

    def test_interpret(self):
        """Tests the result interpretation method."""
        dummy_result = MinimumEigensolverResult()
        dummy_result.eigenvalue = 1.0
        dummy_result.aux_operators_evaluated = {
            "ParticleNumber": (1.0, 0.0),
            "AngularMomentum": (2.0, 0.0),
            "Magnetization": (-1.0, 0.0),
        }

        dummy_problem = ElectronicStructureProblem(
            ElectronicEnergy.from_raw_integrals(np.zeros((2, 2)), np.zeros((2, 2, 2, 2)))
        )
        dummy_problem.hamiltonian.nuclear_repulsion_energy = 1.23
        dummy_problem.reference_energy = -4.56
        dummy_problem.properties.angular_momentum = AngularMomentum(1)
        dummy_problem.properties.magnetization = Magnetization(1)
        dummy_problem.properties.particle_number = ParticleNumber(1)

        elec_struc_res = dummy_problem.interpret(dummy_result)

        with self.subTest("hartree fock energy"):
            self.assertAlmostEqual(elec_struc_res.hartree_fock_energy, -4.56)
        with self.subTest("nuclear repulsion energy"):
            self.assertAlmostEqual(elec_struc_res.nuclear_repulsion_energy, 1.23)
        with self.subTest("computed energy"):
            self.assertEqual(len(elec_struc_res.computed_energies), 1)
            self.assertAlmostEqual(elec_struc_res.computed_energies[0], 1.0)
        with self.subTest("number of particles"):
            self.assertAlmostEqual(elec_struc_res.num_particles[0], 1.0)
        with self.subTest("angular momentum"):
            self.assertAlmostEqual(elec_struc_res.total_angular_momentum[0], 2.0)
        with self.subTest("spin"):
            self.assertAlmostEqual(elec_struc_res.spin[0], 1.0)
        with self.subTest("magnetization"):
            self.assertAlmostEqual(elec_struc_res.magnetization[0], -1.0)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 6
        with open(
            self.get_resource_path("H2_631g_ferm_op.json", "second_q/problems/resources"),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)

        driver = PySCFDriver(basis="631g")
        electronic_structure_problem = driver.run()

        electr_sec_quant_op, second_quantized_ops = electronic_structure_problem.second_q_ops()

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops.values():
                assert isinstance(second_quantized_op, SparseLabelOp)
        with self.subTest("Check components of electronic second quantized operator."):
            assert all(
                s[0] == t[0] and np.isclose(np.abs(s[1]), np.abs(t[1]))
                for s, t in zip(sorted(expected.items()), sorted(electr_sec_quant_op.items()))
            )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_second_q_ops_with_active_space(self):
        """Tests that the correct second quantized operator is created if an active space
        transformer is provided."""
        expected_num_of_sec_quant_ops = 6
        with open(
            self.get_resource_path(
                "H2_631g_ferm_op_active_space.json", "second_q/problems/resources"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)

        driver = PySCFDriver(basis="631g")
        trafo = ActiveSpaceTransformer(2, 2)

        electronic_structure_problem = trafo.transform(driver.run())
        electr_sec_quant_op, second_quantized_ops = electronic_structure_problem.second_q_ops()

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops.values():
                assert isinstance(second_quantized_op, SparseLabelOp)
        with self.subTest("Check components of electronic second quantized operator."):
            assert all(
                s[0] == t[0] and np.isclose(np.abs(s[1]), np.abs(t[1]))
                for s, t in zip(sorted(expected.items()), sorted(electr_sec_quant_op.items()))
            )


if __name__ == "__main__":
    unittest.main()
