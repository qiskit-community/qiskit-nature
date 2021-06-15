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
"""Tests QubitOpBuilder."""
from test import QiskitNatureTestCase
from qiskit.opflow import I, Z, PauliSumOp

from problems.sampling.protein_folding.qubit_op_builder import _create_h_back, \
    _create_h_chiral, _create_h_bbbb, _create_h_bbsc_and_h_scbb, _create_h_scsc, \
    _create_h_contacts, \
    _build_qubit_op, _create_h_short, _create_turn_operators
from problems.sampling.protein_folding.bead_contacts.contact_map import ContactMap
from problems.sampling.protein_folding.bead_distances.distance_map import DistanceMap
from problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import \
    MiyazawaJerniganInteraction
from problems.sampling.protein_folding.penalty_parameters import PenaltyParameters
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


class TestQubitOpBuilder(QiskitNatureTestCase):
    """Tests QubitOpBuilder."""

    def test_check_turns(self):
        """
        Tests that check turns operators are generate correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A"]
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        bead_2 = peptide.get_main_chain[2]
        bead_3 = peptide.get_main_chain[3]
        bead_4 = peptide.get_main_chain[4]
        side_bead_2 = bead_2.side_chain[0]
        side_bead_3 = bead_3.side_chain[0]
        side_bead_4 = bead_4.side_chain[0]

        t_23 = _create_turn_operators(bead_2, bead_3)
        t_34 = _create_turn_operators(bead_3, bead_4)
        t_2s3 = _create_turn_operators(side_bead_2, bead_3)
        t_3s4s = _create_turn_operators(side_bead_3, side_bead_4)
        assert t_23 == 0.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                I) - 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I
                       ^ I) - 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I
                       ^ I)
        assert t_34 == 0.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I)
        assert t_2s3 == 0.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                I) + 0.25 * (
                       I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 0.25 * (
                       I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I)
        assert t_3s4s == 0.25 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                I) + 0.25 * (
                       Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 0.25 * (
                       I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 0.25 * (
                       Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I)

    # TODO
    def test_build_qubit_op(self):
        n_contacts = 0
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        lambda_contacts = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A", "A"]
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", "A", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        penalty_params = PenaltyParameters(lambda_chiral, lambda_back, lambda_1, lambda_contacts)
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        qubit_op = _build_qubit_op(peptide, pair_energies, penalty_params, n_contacts)
        print(qubit_op)

    def test_create_h_back(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        lambda_back = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 0, 0, 0]
        side_chain_residue_sequences = [None, None, None, None, None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        h_back = _create_h_back(peptide, lambda_back)
        assert h_back == 2.5 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) - 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I)

    def test_create_h_back_side_chains(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly in the presence of side
        chains which should not have any influence in this case.
        """
        lambda_back = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        h_back = _create_h_back(peptide, lambda_back)
        assert h_back == 2.5 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) - 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I)

    def test_create_h_chiral(self):
        """
        Tests that the Hamiltonian chirality constraints is created correctly.
        """
        lambda_chiral = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        h_chiral = _create_h_chiral(peptide, lambda_chiral)
        expected = \
            18.75 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            2.5 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z
                    ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            0.625 * (
                    I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            - 1.875 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z
                    ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
            2.5 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
            - 2.5 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                    ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I)
        assert h_chiral == expected

    def test_create_h_bbbb(self):
        lambda_1 = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        x_dist = DistanceMap(peptide)
        contact_map = ContactMap(peptide)
        h_bbbb = _create_h_bbbb(peptide, lambda_1, pair_energies,
                                x_dist, contact_map)
        expected = 4342.5 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 272.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 452.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 195.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 357.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 1067.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 325.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ Z ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 185.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ Z
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 325.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ Z
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 770.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ Z ^ Z
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 325.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ Z ^ Z
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 870.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 280.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 5.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 277.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 180.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 5.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 172.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 22.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 232.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ Z ^ I
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ Z
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 232.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ Z
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 175.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ Z ^ Z
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 232.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ Z ^ Z
                           ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 507.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 5.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 232.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 240.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 195.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 427.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 135.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 190.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 97.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ Z ^ I
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 627.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ Z
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 97.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ Z
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 137.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   - 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ Z ^ Z
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 97.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ Z ^ Z
                           ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 955.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 97.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 232.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 325.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 95.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ I
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 232.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 527.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + \
                   + 575.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 237.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 327.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   + 327.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 530.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 530.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 180.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 240.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 240.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   + 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 235.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 765.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) - 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   + 2.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) - 97.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   + 97.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                           ^ I ^ I
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + 95.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ Z
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 95.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ Z ^ Z
                           ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + 760.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                           ^ I ^ I
                           ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + \
                   - 272.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                           ^ I ^ I
                           ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I)
        assert h_bbbb == expected

    def test_create_h_bbsc_and_h_scbb(self):
        lambda_1 = 10
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A", "A"]
        main_chain_len = 7
        side_chain_lens = [0, 0, 1, 0, 0, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        x_dist = DistanceMap(peptide)

        contact_map = ContactMap(peptide)

        h_bbsc, h_scbb = _create_h_bbsc_and_h_scbb(peptide, lambda_1,
                                                   pair_energies, x_dist,
                                                   contact_map)
        assert h_bbsc == 580.0 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                ^ I ^ I ^ I ^ I) + 165.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 87.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 580.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 165.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 87.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 160.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 160.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 160.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 160.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 5.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 82.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 80.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I)
        assert h_scbb == 515.0 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 87.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 87.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 87.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 515.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 87.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 87.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 87.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z
                       ^ I ^ I ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 85.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z
                       ^ I ^ Z ^ I ^ I ^ I ^ I)

    def test_create_h_bbsc_and_h_scbb_2(self):
        lambda_1 = 10
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A"]
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        x_dist = DistanceMap(peptide)

        contact_map = ContactMap(peptide)

        h_bbsc, h_scbb = _create_h_bbsc_and_h_scbb(peptide, lambda_1,
                                                   pair_energies, x_dist,
                                                   contact_map)
        assert h_bbsc == 767.5 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                I) - 257.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 2.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 172.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 767.5 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 257.5 * (
                       Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 2.5 * (
                       I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 172.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 250.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 85.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 250.0 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 85.0 * (
                       Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 85.0 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 2.5 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 85.0 * (
                       I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 167.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 85.0 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) - 167.5 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 85.0 * (
                       Z ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I) + 170.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I) + 85.0 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I) - 85.0 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I) - 170.0 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I) - 85.0 * (
                       I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I) + 85.0 * (
                       Z ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                       ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I
                       ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I
                       ^ I ^ I) + 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I
                       ^ I ^ I) - 82.5 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I
                       ^ I ^ I)
        assert h_scbb == 0

    def test_create_h_scsc(self):
        lambda_1 = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A", "A"]
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", "A", None]
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        x_dist = DistanceMap(peptide)
        contact_map = ContactMap(peptide)
        h_scsc = _create_h_scsc(peptide, lambda_1,
                                pair_energies, x_dist, contact_map)
        assert h_scsc == 920.0 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 105.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 105.0 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 105.0 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 920.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 105.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 105.0 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 105.0 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ Z ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z
                       ^ Z ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ Z ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 102.5 * (
                       Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 102.5 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z
                       ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) + 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I)

    def test_create_h_short(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A", "A", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        h_short = _create_h_short(peptide, pair_energies).reduce()
        expected = PauliSumOp.from_list([("IIIIIIIIIIIIIIIIIIIIIIIIIIII", 0)])
        assert h_short == expected

    #
    # def test_create_h_short_old(self):
    #     """
    #         Tests that the Hamiltonian to back-overlaps is created correctly.
    #         """
    #     lf = LatticeFoldingProblem(residue_sequence=["A", "P", "R", "L", "A", "A", "A"])
    #     lf.pauli_op()
    #     N = 8
    #     side_chain = [0, 0, 1, 1, 1, 1, 1, 0]
    #     pair_energies = lf._pair_energies
    #
    #     pauli_conf = _create_pauli_for_conf(N)
    #     qubits = _create_qubits_for_conf(pauli_conf)
    #     indic_0, indic_1, indic_2, indic_3, n_conf = _create_indic_turn(N, side_chain, qubits)
    #     delta_n0, delta_n1, delta_n2, delta_n3 = _create_delta_BB(N, indic_0, indic_1, indic_2,
    #                                                               indic_3, pauli_conf)
    #     delta_n0, delta_n1, delta_n2, delta_n3 = _add_delta_SC(N, delta_n0, delta_n1, delta_n2,
    #                                                            delta_n3, indic_0, indic_1,
    #                                                            indic_2,
    #                                                            indic_3, pauli_conf)
    #     x_dist = _create_x_dist(N, delta_n0, delta_n1, delta_n2, delta_n3, pauli_conf)
    #
    #     h_short = _create_H_short(N, side_chain, pair_energies,
    #                               x_dist, pauli_conf, indic_0,
    #                               indic_1, indic_2, indic_3)
    #     print(h_short)

    def test_create_h_contacts(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        lambda_contacts = 10
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)

        n_contacts = 0
        contact_map = ContactMap(peptide)
        h_contacts = _create_h_contacts(peptide, contact_map, lambda_contacts, n_contacts)
        assert h_contacts == 280.0 * (
                I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 50.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 50.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 10.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 100.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 20.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 10.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 50.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 10.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 5.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 50.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 5.0 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 10.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                       I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I
                       ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I)
