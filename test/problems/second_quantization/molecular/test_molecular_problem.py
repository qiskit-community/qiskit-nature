# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests Molecular Problem."""

from test import QiskitNatureTestCase
from test.problems.second_quantization.molecular.resources.resource_reader import read_expected_file
import numpy as np
from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.molecular.molecular_problem import MolecularProblem


class TestMolecularProblem(QiskitNatureTestCase):
    """Tests Molecular Problem."""

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 7
        expected_fermionic_op_path = self.get_resource_path('H2_631g_ferm_op_two_ints',
                                                            'problems/second_quantization/'
                                                            'molecular/resources')
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)

        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        molecular_problem = MolecularProblem(driver)

        second_quantized_ops = molecular_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops[0]
        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops:
                assert isinstance(second_quantized_op, SecondQuantizedOp)
        with self.subTest("Check components of electronic second quantized operator."):
            assert all(s[0] == t[0] and np.isclose(s[1], t[1]) for s, t in
                       zip(expected_fermionic_op, electr_sec_quant_op.to_list()))
        # TODO test QMolecule itself if it is ever a field in MolecularProblem

    def test_second_q_ops_with_active_space(self):
        """Tests that the correct second quantized operator is created if an active space
        transformer is provided."""
        expected_num_of_sec_quant_ops = 4
        expected_fermionic_op_path = self.get_resource_path('H2_631g_ferm_op_active_space',
                                                            'problems/second_quantization/'
                                                            'molecular/resources')
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        trafo = ActiveSpaceTransformer(num_electrons=2, num_orbitals=2)

        molecular_problem = MolecularProblem(driver, [trafo])
        second_quantized_ops = molecular_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops[0]

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops:
                assert isinstance(second_quantized_op, SecondQuantizedOp)
        with self.subTest("Check components of electronic second quantized operator."):
            assert all(s[0] == t[0] and np.isclose(s[1], t[1]) for s, t in
                       zip(expected_fermionic_op, electr_sec_quant_op.to_list()))
        # TODO test QMolecule itself if it is ever a field in MolecularProblem
