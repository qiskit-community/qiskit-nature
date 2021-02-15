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

from test import QiskitNatureTestCase
from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.molecular.molecular_problem import MolecularProblem


class TestMolecularProblem(QiskitNatureTestCase):
    """Molecular Problem tests."""

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.595',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
            transformation = []
            self.molecular_problem = MolecularProblem(self.driver, transformation)

        except QiskitNatureError:
            self.skipTest('PYSCF driver does not appear to be installed')

    def test_second_q_ops(self):
        expected_num_of_electr_terms = 631
        expected_num_of_sec_quant_ops = 7

        second_quantized_ops = self.molecular_problem.second_q_ops()
        electronic_second_quantized_op = second_quantized_ops[0]
        assert electronic_second_quantized_op.boson is None
        assert electronic_second_quantized_op.spin == {}
        assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        for second_quantized_op in second_quantized_ops:
            assert isinstance(second_quantized_op, SecondQuantizedOp)
        assert len(electronic_second_quantized_op.fermion) == expected_num_of_electr_terms
