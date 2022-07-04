# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Bravyi-Kitaev Super-Fast Mapper """

import unittest
from test import QiskitNatureTestCase

import numpy as np
from qiskit.quantum_info import SparsePauliOp, PauliList

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevSuperFastMapper
from qiskit_nature.second_q.mappers.bksf import (
    _edge_operator_aij,
    _edge_operator_bi,
    _bksf_edge_list_fermionic_op,
)

from .resources.bksf_lih import (
    FERMIONIC_HAMILTONIAN,
    QUBIT_HAMILTONIAN,
)


def _sort_simplify(sparse_pauli):
    sparse_pauli = sparse_pauli.simplify()
    indices = sparse_pauli.paulis.argsort()
    table = sparse_pauli.paulis[indices]
    coeffs = sparse_pauli.coeffs[indices]
    sparse_pauli = SparsePauliOp(table, coeffs)
    return sparse_pauli


class TestBravyiKitaevSuperFastMapper(QiskitNatureTestCase):
    """Test Bravyi-Kitaev Super-Fast Mapper"""

    def test_bksf_edge_op_bi(self):
        """Test BKSF mapping, edge operator bi"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_b0 = _edge_operator_bi(edge_list, 0)
        qterm_b1 = _edge_operator_bi(edge_list, 1)
        qterm_b2 = _edge_operator_bi(edge_list, 2)
        qterm_b3 = _edge_operator_bi(edge_list, 3)

        ref_qterm_b0 = SparsePauliOp("IIIZZZ")
        ref_qterm_b1 = SparsePauliOp("IZZIIZ")
        ref_qterm_b2 = SparsePauliOp("ZIZIZI")
        ref_qterm_b3 = SparsePauliOp("ZZIZII")

        with self.subTest("Test edge operator b0"):
            self.assertEqual(qterm_b0, ref_qterm_b0)
        with self.subTest("Test edge operator b1"):
            self.assertEqual(qterm_b1, ref_qterm_b1)
        with self.subTest("Test edge operator b2"):
            self.assertEqual(qterm_b2, ref_qterm_b2)
        with self.subTest("Test edge operator b3"):
            self.assertEqual(qterm_b3, ref_qterm_b3)

    def test_bksf_edge_op_aij(self):
        """Test BKSF mapping, edge operator aij"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_a01 = _edge_operator_aij(edge_list, 0, 1)
        qterm_a02 = _edge_operator_aij(edge_list, 0, 2)
        qterm_a03 = _edge_operator_aij(edge_list, 0, 3)
        qterm_a12 = _edge_operator_aij(edge_list, 1, 2)
        qterm_a13 = _edge_operator_aij(edge_list, 1, 3)
        qterm_a23 = _edge_operator_aij(edge_list, 2, 3)

        ref_qterm_a01 = SparsePauliOp("IIIIIX")
        ref_qterm_a02 = SparsePauliOp("IIIIXZ")
        ref_qterm_a03 = SparsePauliOp("IIIXZZ")
        ref_qterm_a12 = SparsePauliOp("IIXIZZ")
        ref_qterm_a13 = SparsePauliOp("IXZZIZ")
        ref_qterm_a23 = SparsePauliOp("XZZZZI")

        with self.subTest("Test edge operator a01"):
            self.assertEqual(qterm_a01, ref_qterm_a01)
        with self.subTest("Test edge operator a02"):
            self.assertEqual(qterm_a02, ref_qterm_a02)
        with self.subTest("Test edge operator a03"):
            self.assertEqual(qterm_a03, ref_qterm_a03)
        with self.subTest("Test edge operator a12"):
            self.assertEqual(qterm_a12, ref_qterm_a12)
        with self.subTest("Test edge operator a13"):
            self.assertEqual(qterm_a13, ref_qterm_a13)
        with self.subTest("Test edge operator a23"):
            self.assertEqual(qterm_a23, ref_qterm_a23)

    def test_h2(self):
        """Test H2 molecule"""
        with self.subTest("Excitation edges 1"):
            assert np.alltrue(
                _bksf_edge_list_fermionic_op(FermionicOp("+-+-", display_format="dense"))
                == np.array([[0, 1], [2, 3]])
            )

        with self.subTest("Excitation edges 2"):
            assert np.alltrue(
                _bksf_edge_list_fermionic_op(FermionicOp("+--+", display_format="dense"))
                == np.array([[0, 1], [3, 2]])
            )

        ## H2 from pyscf with sto-3g basis
        h2_fop = FermionicOp(
            [
                ("+-+-", (0.18128880821149607 + 0j)),
                ("+--+", (-0.18128880821149607 + 0j)),
                ("-++-", (-0.18128880821149607 + 0j)),
                ("-+-+", (0.18128880821149604 + 0j)),
                ("IIIN", (-0.4759487152209648 + 0j)),
                ("IINI", (-1.2524635735648986 + 0j)),
                ("IINN", (0.48217928821207245 + 0j)),
                ("INII", (-0.4759487152209648 + 0j)),
                ("ININ", (0.697393767423027 + 0j)),
                ("INNI", (0.6634680964235684 + 0j)),
                ("NIII", (-1.2524635735648986 + 0j)),
                ("NIIN", (0.6634680964235684 + 0j)),
                ("NINI", (0.6744887663568382 + 0j)),
                ("NNII", (0.48217928821207245 + 0j)),
            ],
            display_format="dense",
        )

        expected_pauli_op = SparsePauliOp.from_list(
            [
                ("IIII", (-0.8126179630230767 + 0j)),
                ("IIZZ", (0.17119774903432952 + 0j)),
                ("IYYI", (0.04532220205287402 + 0j)),
                ("IZIZ", (0.17119774903432955 + 0j)),
                ("IZZI", (0.34297063344496626 + 0j)),
                ("XIIX", (0.04532220205287402 + 0j)),
                ("YIIY", (0.04532220205287402 + 0j)),
                ("YZZY", (0.04532220205287402 + 0j)),
                ("ZIIZ", (0.3317340482117842 + 0j)),
                ("ZIZI", (-0.22278593040418454 + 0j)),
                ("ZXXZ", (0.04532220205287402 + 0j)),
                ("ZYYZ", (0.04532220205287402 + 0j)),
                ("ZZII", (-0.22278593040418454 + 0j)),
                ("ZZZZ", (0.24108964410603623 + 0j)),
            ]
        )

        pauli_sum_op = BravyiKitaevSuperFastMapper().map(h2_fop)

        op1 = _sort_simplify(expected_pauli_op)
        op2 = _sort_simplify(pauli_sum_op.primitive)

        with self.subTest("Map H2 frome sto3g basis, number of terms"):
            self.assertEqual(len(op1), len(op2))

        with self.subTest("Map H2 frome sto3g basis result"):
            self.assertEqual(op1, op2)

        with self.subTest("Sparse FermionicOp input"):
            h2_fop_sparse = h2_fop.normal_ordered()
            pauli_sum_op_from_sparse = BravyiKitaevSuperFastMapper().map(h2_fop_sparse)
            op2_from_sparse = _sort_simplify(pauli_sum_op_from_sparse.primitive)
            self.assertEqual(op1, op2_from_sparse)

        with self.subTest("Test accepting identity with zero coefficient"):
            h2_fop_zero_term = h2_fop + FermionicOp([("IIII", 0.0)], display_format="dense")
            pauli_sum_op_extra = BravyiKitaevSuperFastMapper().map(h2_fop_zero_term)
            op3 = _sort_simplify(pauli_sum_op_extra.primitive)
            self.assertEqual(op1, op3)

        with self.subTest("Test zero FermiOp"):
            fermi_op = FermionicOp([("++--", 0.0)], display_format="dense")
            pauli_op = BravyiKitaevSuperFastMapper().map(fermi_op).primitive
            x = np.array([], dtype="bool")
            z = np.array([], dtype="bool")
            expected_pauli_op = SparsePauliOp(PauliList.from_symplectic(x, z), coeffs=[0.0])
            self.assertEqual(pauli_op, expected_pauli_op)

    def test_LiH(self):
        """Test LiH molecule"""
        pauli_sum_op = BravyiKitaevSuperFastMapper().map(FERMIONIC_HAMILTONIAN)
        self.assertEqual(pauli_sum_op._primitive, QUBIT_HAMILTONIAN)


if __name__ == "__main__":
    unittest.main()
