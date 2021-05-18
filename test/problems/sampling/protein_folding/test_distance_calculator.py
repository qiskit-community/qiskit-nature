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
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from problems.sampling.protein_folding.distance_calculator import _calc_distances_main_chain, \
    _add_distances_side_chain, \
    _calc_total_distances
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from test import QiskitNatureTestCase


class TestDistanceCalculator(QiskitNatureTestCase):
    """Tests DistanceCalculator."""

    def setUp(self):
        super().setUp()
        main_chain_residue_seq = "SAA"
        main_chain_len = 3
        side_chain_lens = [0, 0, 1]
        side_chain_residue_sequences = [None, None, "A"]

        self.peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                               side_chain_residue_sequences)

    def test_calc_distances_main_chain(self):
        """
        Tests that distances for all beads on the main chain are calculated correctly.
        """
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(self.peptide)
        # TODO refactor once a better data structure for distances is created
        assert delta_n0 == {1: {0: {2: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False],
                                         [False, False, False, False, False, False, True, False],
                                         [False, False, False, False, False, True, False, False],
                                         [False, False, False, False, False, True, True, False]],
                                        coeffs=[-0.25 + 0.j, -0.25 + 0.j, -0.25 + 0.j,
                                                -0.25 + 0.j]), coeff=1.0)}, 3: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, True],
                                         [False, False, False, False, False, False, True, True],
                                         [False, False, False, False, False, True, False, False],
                                         [False, False, False, False, False, True, True, False]],
                                        coeffs=[0.25 + 0.j, 0.25 + 0.j, -0.25 + 0.j, -0.25 + 0.j]),
                          coeff=1.0)}}, 1: {2: {}, 3: {}}}, 2: {0: {3: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False],
                                         [False, False, False, False, False, False, False, True],
                                         [False, False, False, False, False, False, True, False],
                                         [False, False, False, False, False, False, True, True]],
                                        coeffs=[0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j]),
                          coeff=1.0)}}, 1: {3: {}}}}

    # TODO should we observe different values of deltas than in main chain?
    def test_add_distances_side_chain(self):
        """
        Tests that distances for all beads on side chains are calculated correctly.
        """
        delta_n0_main, delta_n1_main, delta_n2_main, delta_n3_main = _calc_distances_main_chain(
            self.peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(self.peptide,
                                                                           delta_n0_main,
                                                                           delta_n1_main,
                                                                           delta_n2_main,
                                                                           delta_n3_main)
        assert delta_n0 == {1: {0: {2: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False],
                                         [False, False, False, False, False, False, True, False],
                                         [False, False, False, False, False, True, False, False],
                                         [False, False, False, False, False, True, True, False]],
                                        coeffs=[-0.25 + 0.j, -0.25 + 0.j, -0.25 + 0.j,
                                                -0.25 + 0.j]), coeff=1.0)}, 3: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, True],
                                         [False, False, False, False, False, False, True, True],
                                         [False, False, False, False, False, True, False, False],
                                         [False, False, False, False, False, True, True, False]],
                                        coeffs=[0.25 + 0.j, 0.25 + 0.j, -0.25 + 0.j, -0.25 + 0.j]),
                          coeff=1.0)}}, 1: {2: {}, 3: {}}}, 2: {0: {3: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False],
                                         [False, False, False, False, False, False, False, True],
                                         [False, False, False, False, False, False, True, False],
                                         [False, False, False, False, False, False, True, True]],
                                        coeffs=[0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j, 0.25 + 0.j]),
                          coeff=1.0)}}, 1: {3: {}}}}

    def test_calc_total_distances(self):
        """
        Tests that total distances for all beads are calculated correctly.
        """
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(self.peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(self.peptide, delta_n0,
                                                                           delta_n1,
                                                                           delta_n2, delta_n3)
        x_dist = _calc_total_distances(self.peptide, delta_n0, delta_n1, delta_n2, delta_n3)
        assert x_dist == {1: {0: {2: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False]],
                                        coeffs=[1. + 0.j]), coeff=1.0)}, 3: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False],
                                         [False, False, False, False, False, False, True, True],
                                         [False, False, False, False, False, True, False, True],
                                         [False, False, False, False, False, True, True, False]],
                                        coeffs=[1.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j]),
                          coeff=1.0)}}, 1: {2: {}, 3: {}}}, 2: {0: {3: {
            0: PauliSumOp(SparsePauliOp([[False, False, False, False, False, False, False, False]],
                                        coeffs=[1. + 0.j]), coeff=1.0)}}, 1: {3: {}}}}
