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

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.operator_factories.lattice import FermiHubbardModel
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.operator_factories.lattices import (
    BoundaryCondition,
    LineLattice,
)
from qiskit_nature.second_q.problems import EigenstateResult


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
        self._compare_second_q_op(fhm.second_q_ops(), lmp.second_q_ops()[lmp.main_property_name])

    def test_interpret(self):
        """Tests that the result is interpreted"""
        num_nodes = 4
        boundary_condition = BoundaryCondition.OPEN
        line_lattice = LineLattice(num_nodes=num_nodes, boundary_condition=boundary_condition)
        fhm = FermiHubbardModel(lattice=line_lattice, onsite_interaction=5.0)
        eigenenergies = np.array([-1])
        eigenstates = [np.array([1, 0])]
        aux_operator_eigenvalues = [(1, 2)]
        # For EigenstateResult
        lmp = LatticeModelProblem(fhm)
        eigenstate_result = EigenstateResult()
        eigenstate_result.eigenenergies = eigenenergies
        eigenstate_result.eigenstates = eigenstates
        eigenstate_result.aux_operator_eigenvalues = aux_operator_eigenvalues
        lmr = lmp.interpret(eigenstate_result)
        self.assertEqual(lmr.eigenenergies, eigenstate_result.eigenenergies)
        self.assertEqual(lmr.eigenstates, eigenstate_result.eigenstates)
        self.assertEqual(lmr.aux_operator_eigenvalues, eigenstate_result.aux_operator_eigenvalues)
        # For EigenSOlverResult
        lmp = LatticeModelProblem(fhm)
        eigensolver_result = EigensolverResult()
        eigensolver_result.eigenvalues = eigenenergies
        eigensolver_result.eigenstates = eigenstates
        eigensolver_result.aux_operator_eigenvalues = [aux_operator_eigenvalues]
        lmr = lmp.interpret(eigensolver_result)
        self.assertEqual(lmr.eigenenergies, eigensolver_result.eigenvalues)
        self.assertEqual(lmr.eigenstates, eigensolver_result.eigenstates)
        self.assertEqual(lmr.aux_operator_eigenvalues, eigensolver_result.aux_operator_eigenvalues)
        # For MinimumEigensolverResult
        lmp = LatticeModelProblem(fhm)
        mes_result = MinimumEigensolverResult()
        mes_result.eigenvalue = -1
        mes_result.eigenstate = np.array([1, 0])
        mes_result.aux_operator_eigenvalues = aux_operator_eigenvalues
        lmr = lmp.interpret(mes_result)
        self.assertEqual(lmr.eigenenergies, np.asarray([mes_result.eigenvalue]))
        self.assertEqual(lmr.eigenstates, [mes_result.eigenstate])
        self.assertEqual(lmr.aux_operator_eigenvalues, [mes_result.aux_operator_eigenvalues])
