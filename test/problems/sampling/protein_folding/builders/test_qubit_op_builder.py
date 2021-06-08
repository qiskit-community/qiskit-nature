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
from qiskit.opflow import PauliOp, I, Z

from problems.sampling.protein_folding.builders.qubit_op_builder import _create_h_back, \
    _create_h_chiral, _create_h_bbbb, _create_h_bbsc_and_h_scbb, _create_h_scsc, _create_H_contacts
from problems.sampling.protein_folding.contact_map import ContactMap
from problems.sampling.protein_folding.distance_calculator import _calc_distances_main_chain, \
    _add_distances_side_chain, _calc_total_distances
from problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import \
    MiyazawaJerniganInteraction
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from test import QiskitNatureTestCase


class TestContactQubitsBuilder(QiskitNatureTestCase):
    """Tests ContactQubitsBuilder."""

    def test_check_turns(self) -> PauliOp:
        """

        """
        pass

    def test_create_h_back(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        lambda_back = 10
        main_chain_residue_seq = "SAASS"
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
        main_chain_residue_seq = "SAASS"
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
        main_chain_residue_seq = "SAASSASA"
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

    def test_create_H_BBBB(self):
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        main/backbone (BB) beads

        Args:
            main_chain_len: Number of total beads in peptide
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact
            pair_energies: Numpy array of pair energies for amino acids
            x_dist: Numpy array that tracks all distances between backbone and side chain
                    beads for all axes: 0,1,2,3
            pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
            contacts: Dictionary of contact qubits in symbolic notation

        Returns:
            H_BBBB: Hamiltonian term in symbolic notation
        """
        lambda_1 = 10
        main_chain_residue_seq = "SAASSASA"
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        mj = MiyazawaJerniganInteraction()
        pair_energies = mj.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        contact_map = ContactMap(peptide)
        h_bbbb = _create_h_bbbb(main_chain_len, lambda_1, pair_energies,
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

    def test_create_H_BBSC_and_H_SCBB(self):
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        main/backbone (BB) and side chain (SC) beads. In the absence
        of side chains, this function returns a value of 0.

        Args:
            main_chain_len: Number of total beads in peptide
            side: List of side chains in peptide
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact
            pair_energies: Numpy array of pair energies for amino acids
            x_dist: Numpy array that tracks all distances between backbone and side chain
                    beads for all axes: 0,1,2,3
            pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
            contacts: Dictionary of contact qubits in symbolic notation

        Returns:
            H_BBSC, H_SCBB: Tuple of Hamiltonian terms consisting of backbone and side chain
            interactions
        """

        lambda_1 = 10
        main_chain_residue_seq = 'APRLRAA'
        main_chain_len = 7
        side_chain_lens = [0, 0, 1, 0, 0, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", None]
        mj = MiyazawaJerniganInteraction()
        pair_energies = mj.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        side_chain = peptide.get_side_chain_hot_vector()
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)

        contact_map = ContactMap(peptide)

        H_BBSC, H_SCBB = _create_h_bbsc_and_h_scbb(main_chain_len, side_chain, lambda_1,
                                                   pair_energies, x_dist,
                                                   contact_map)
        assert H_BBSC == 580.0 * (
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
        assert H_SCBB == 515.0 * (
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

    def test_create_H_BBSC_and_H_SCBB_2(self):
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        main/backbone (BB) and side chain (SC) beads. In the absence
        of side chains, this function returns a value of 0.

        Args:
            main_chain_len: Number of total beads in peptide
            side: List of side chains in peptide
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact
            pair_energies: Numpy array of pair energies for amino acids
            x_dist: Numpy array that tracks all distances between backbone and side chain
                    beads for all axes: 0,1,2,3
            pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
            contacts: Dictionary of contact qubits in symbolic notation

        Returns:
            H_BBSC, H_SCBB: Tuple of Hamiltonian terms consisting of backbone and side chain
            interactions
        """

        lambda_1 = 10
        main_chain_residue_seq = 'APRLAA'
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", None]
        mj = MiyazawaJerniganInteraction()
        pair_energies = mj.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        side_chain = peptide.get_side_chain_hot_vector()
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)

        contact_map = ContactMap(peptide)

        H_BBSC, H_SCBB = _create_h_bbsc_and_h_scbb(main_chain_len, side_chain, lambda_1,
                                                   pair_energies, x_dist,
                                                   contact_map)
        assert H_BBSC == 767.5 * (
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
        assert H_SCBB == 0

    def test_create_H_SCSC(self):
        """
            Creates Hamiltonian term corresponding to 1st neighbor interaction between
            side chain (SC) beads. In the absence of side chains, this function
            returns a value of 0.

            Args:
                main_chain_len: Number of total beads in peptides
                lambda_1: Constraint to penalize local overlap between
                         beads within a nearest neighbor contact
                pair_energies: Numpy array of pair energies for amino acids
                x_dist: Numpy array that tracks all distances between backbone and side chain
                        beads for all axes: 0,1,2,3
                pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
                contacts: Dictionary of contact qubits in symbolic notation

            Returns:
                H_SCSC: Hamiltonian term consisting of side chain pairwise interactions
            """
        lambda_1 = 10
        main_chain_residue_seq = "SAASSASAA"
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", "A", None]
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        mj = MiyazawaJerniganInteraction()
        pair_energies = mj.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        side_chain = peptide.get_side_chain_hot_vector()
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        contact_map = ContactMap(peptide)
        H_SCSC = _create_h_scsc(main_chain_len, side_chain, lambda_1,
                                pair_energies, x_dist, contact_map)
        assert H_SCSC == 920.0 * (
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

    # def test_create_h_short(self):
    #     """
    #         Tests that the Hamiltonian to back-overlaps is created correctly.
    #         """
    #     main_chain_residue_seq = 'APRLRAAA'
    #     main_chain_len = 8
    #     side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
    #     side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
    #     mj = MiyazawaJerniganInteraction()
    #     pair_energies = mj.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
    #     peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
    #                       side_chain_residue_sequences)
    #     h_short = _create_h_short(peptide, pair_energies)
    #     print(h_short)
    #
    # def test_create_h_short_old(self):
    #     """
    #         Tests that the Hamiltonian to back-overlaps is created correctly.
    #         """
    #     lf = LatticeFoldingProblem(residue_sequence="SAASSASSSS")
    #     lf.pauli_op()
    #     N = 10
    #     side_chain = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]
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
        main_chain_residue_seq = 'APRLRAAA'
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)

        N_contacts = 0
        contact_map = ContactMap(peptide)
        h_contacts = _create_H_contacts(peptide, contact_map, lambda_contacts, N_contacts)
        print(h_contacts)
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
                           Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 5.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) - 10.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + 10.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I)
