# This code is part of Qiskit.
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
import warnings
from test import QiskitNatureTestCase

import json
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.opflow.primitive_ops import Z2Symmetries
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries as Z2SparseSymmetries

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


class TestElectronicStructureProblem(QiskitNatureTestCase):
    """Tests Electronic Structure Problem."""

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

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_symmetry_sector_locator(self):
        """Tests that the symmetry sector locator gives the right sector."""
        driver = PySCFDriver()
        electronic_structure_problem = driver.run()
        hamiltonian, _ = electronic_structure_problem.second_q_ops()
        mapper = JordanWignerMapper()
        mapped_op = mapper.map(hamiltonian)
        expected_sector = [-1, 1, -1]

        with self.subTest("Opflow Z2Symmetries"):
            if isinstance(mapped_op, PauliSumOp):
                mapped_op = mapped_op.primitive
            z2sym = Z2SparseSymmetries.find_z2_symmetries(mapped_op)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                sector = electronic_structure_problem.symmetry_sector_locator(z2sym, mapper)
            self.assertEqual(sector, expected_sector)
        with self.subTest("Opflow Z2Symmetries"):
            if not isinstance(mapped_op, PauliSumOp):
                mapped_op = PauliSumOp(mapped_op)
            z2sym = Z2Symmetries.find_Z2_symmetries(mapped_op)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                sector = electronic_structure_problem.symmetry_sector_locator(z2sym, mapper)
            self.assertEqual(sector, expected_sector)


if __name__ == "__main__":
    unittest.main()
