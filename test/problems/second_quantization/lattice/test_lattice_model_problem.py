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

"""Tests Lattice Model Problem."""

from test import QiskitNatureTestCase

from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import LatticeModelProblem
from qiskit_nature.problems.second_quantization.lattice import (
    BoundaryCondition,
    FermiHubbardModel,
    LineLattice,
)


class TestLatticeModelProblem(QiskitNatureTestCase):
    """Tests Lattice Model Problem."""

    def _compare_second_q_op(self, first: SecondQuantizedOp, second: SecondQuantizedOp):
        """Compares second quantized operators"""
        f_list = first.to_list()
        s_list = second.to_list()
        self.assertEqual(len(f_list), len(s_list))
        for f_term, s_term in zip(f_list, s_list):
            # compare labels
            self.assertEqual(f_term[0], s_term[0])
            # compare coefficients
            self.assertEqual(f_term[1], s_term[1])

    def test_second_q_ops(self):
        """Tests that the list of second quantized operators is created."""
        num_nodes = 4
        boundary_condition = BoundaryCondition.OPEN
        line_lattice = LineLattice(num_nodes=num_nodes, boundary_condition=boundary_condition)
        fhm = FermiHubbardModel(lattice=line_lattice, onsite_interaction=5.0)
        lmp = LatticeModelProblem(fhm)
        self._compare_second_q_op(fhm.second_q_ops(), lmp.second_q_ops()[0])
