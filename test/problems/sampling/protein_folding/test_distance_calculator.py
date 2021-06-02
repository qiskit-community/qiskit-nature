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
from qiskit.opflow import I, Z

from problems import LatticeFoldingProblem
from problems.sampling.folding.folding_qubit_op_builder import _create_pauli_for_conf, \
    _create_qubits_for_conf, _create_indic_turn
from problems.sampling.protein_folding.distance_calculator import _calc_distances_main_chain, \
    _add_distances_side_chain, \
    _calc_total_distances
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from test import QiskitNatureTestCase


class TestDistanceCalculator(QiskitNatureTestCase):
    """Tests DistanceCalculator."""

    def setUp(self):
        super().setUp()
        main_chain_residue_seq = "SAAAA"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]

        self.peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                               side_chain_residue_sequences)

        lf = LatticeFoldingProblem('SAAAA')
        lf.pauli_op()
        self.N = 5
        side_chain = [0, 0, 1, 0, 0]
        # pair_energies = lf._pair_energies

        self.pauli_conf = _create_pauli_for_conf(self.N)
        qubits = _create_qubits_for_conf(self.pauli_conf)
        self.indic_0, self.indic_1, self.indic_2, self.indic_3, self.n_conf = _create_indic_turn(
            self.N, side_chain, qubits)

    def test_calc_distances_main_chain(self):
        """
        Tests that distances for all beads on the main chain are calculated correctly.
        """
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(self.peptide)
        # checking only some of the entries
        assert delta_n0[1][0][5][0] == 1.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I)
        assert delta_n0[2][0][5][0] == 1.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I)
        assert delta_n1[1][0][4][0] == -1.5 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I)
        assert delta_n1[1][0][5][0] == -1.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 0.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I)


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

        assert delta_n0[1][0][3][1] == 0.75 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                       I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                       I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                       I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I)


    def test_calc_total_distances(self):
        """
        Tests that total distances for all beads are calculated correctly.
        """
        delta_n0_main, delta_n1_main, delta_n2_main, delta_n3_main = _calc_distances_main_chain(
            self.peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(self.peptide,
                                                                           delta_n0_main,
                                                                           delta_n1_main,
                                                                           delta_n2_main,
                                                                           delta_n3_main)
        x_dist = _calc_total_distances(self.peptide, delta_n0, delta_n1, delta_n2, delta_n3)
        assert x_dist[2][0][3][1] == 1.5 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.5 * (
                           I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.5 * (
                           I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.5 * (
                           I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I)
        assert x_dist[1][0][3][1] == 3.0 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 1.0 * (
                           I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 1.0 * (
                           I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I)

