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
from qiskit_nature.problems.second_quantization.molecular.fermionic_op_builder import \
    build_fermionic_op
from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.drivers import HDF5Driver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.molecular.molecular_problem import MolecularProblem


class TestMolecularProblem(QiskitNatureTestCase):
    """Tests Molecular Problem."""

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_electr_terms = 185
        expected_num_of_sec_quant_ops = 7

        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        molecular_problem = MolecularProblem(driver)

        expected_electr_sec_quant_op = TestMolecularProblem._calc_expected_sec_quant_op(driver,
                                                                                        None)

        second_quantized_ops = molecular_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops[0]
        assert electr_sec_quant_op.boson is None
        assert electr_sec_quant_op.spin == {}
        assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        for second_quantized_op in second_quantized_ops:
            assert isinstance(second_quantized_op, SecondQuantizedOp)
        assert len(electr_sec_quant_op.fermion) == expected_num_of_electr_terms
        assert expected_electr_sec_quant_op.__eq__(electr_sec_quant_op)

    def test_second_q_ops_with_active_space(self):
        """Tests that the correct second quantized operator is created if an active space
        transformer is provided."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        trafo = ActiveSpaceTransformer(num_electrons=2, num_orbitals=2)
        expected_electr_sec_quant_op = TestMolecularProblem._calc_expected_sec_quant_op(driver,
                                                                                        trafo)

        molecular_problem = MolecularProblem(driver, [trafo])
        second_quantized_ops_list = molecular_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops_list[0]

        assert expected_electr_sec_quant_op.__eq__(electr_sec_quant_op)
        # TODO test QMolecule itself if it is ever a field in MolecularProblem

    @staticmethod
    def _calc_expected_sec_quant_op(driver, trafo):
        q_molecule = driver.run()
        if trafo:
            q_molecule = trafo.transform(q_molecule)
        electronic_fermionic_op = build_fermionic_op(q_molecule)
        return SecondQuantizedOp([electronic_fermionic_op])
