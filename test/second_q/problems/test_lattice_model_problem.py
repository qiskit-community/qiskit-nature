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

import unittest
from test import QiskitNatureTestCase

import numpy as np

from qiskit.algorithms.eigensolvers import EigensolverResult
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LineLattice,
)
from qiskit_nature.second_q.problems import EigenstateResult


class TestLatticeModelProblem(QiskitNatureTestCase):
    """Tests Lattice Model Problem."""

    def _compare_second_q_op(self, first: SecondQuantizedOp, second: SecondQuantizedOp):
        """Compares second quantized operators"""
        f_list = first.items()
        s_list = second.items()
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
        expected_op = fhm.second_q_op()
        lmp = LatticeModelProblem(fhm)
        main_op, _ = lmp.second_q_ops()
        self._compare_second_q_op(expected_op, main_op)

    def test_interpret(self):
        """Tests that the result is interpreted"""
        num_nodes = 4
        boundary_condition = BoundaryCondition.OPEN
        line_lattice = LineLattice(num_nodes=num_nodes, boundary_condition=boundary_condition)
        fhm = FermiHubbardModel(lattice=line_lattice, onsite_interaction=5.0)
        eigenvalues = np.array([-1])
        aux_operators_evaluated = [[1, 2]]
        # For EigenstateResult
        lmp = LatticeModelProblem(fhm)
        eigenstate_result = EigenstateResult()
        eigenstate_result.eigenvalues = eigenvalues
        eigenstate_result.aux_operators_evaluated = aux_operators_evaluated
        lmr = lmp.interpret(eigenstate_result)
        self.assertEqual(lmr.eigenvalues, eigenstate_result.eigenvalues)
        self.assertEqual(lmr.aux_operators_evaluated, eigenstate_result.aux_operators_evaluated)
        # For EigenSOlverResult
        lmp = LatticeModelProblem(fhm)
        eigensolver_result = EigensolverResult()
        eigensolver_result.eigenvalues = eigenvalues
        eigensolver_result.aux_operators_evaluated = [[(1, {}), (2, {})]]
        lmr = lmp.interpret(eigensolver_result)
        self.assertEqual(lmr.eigenvalues, eigensolver_result.eigenvalues)
        self.assertEqual(lmr.aux_operators_evaluated, aux_operators_evaluated)
        # For MinimumEigensolverResult
        lmp = LatticeModelProblem(fhm)
        mes_result = MinimumEigensolverResult()
        mes_result.eigenvalue = -1
        mes_result.eigenstate = np.array([1, 0])
        mes_result.aux_operators_evaluated = [(1, {}), (2, {})]
        lmr = lmp.interpret(mes_result)
        self.assertEqual(lmr.eigenvalues, np.asarray([mes_result.eigenvalue]))
        self.assertEqual(lmr.aux_operators_evaluated, aux_operators_evaluated)


if __name__ == "__main__":
    unittest.main()
