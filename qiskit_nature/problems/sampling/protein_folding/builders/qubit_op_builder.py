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
from qiskit.opflow import PauliOp, I

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


def _build_qubit_op(n, side_chain, pair_energies, lambda_chiral, lambda_back, lambda_1,
                    lambda_contacts, n_contacts):
    pass


def _check_turns(lower_bead: BaseBead, higher_bead: BaseBead) -> PauliOp:
    lower_bead_indic_0, lower_bead_indic_1, lower_bead_indic_2, lower_bead_indic_3 = \
        lower_bead.get_indicator_functions()

    higher_bead_indic_0, higher_bead_indic_1, higher_bead_indic_2, higher_bead_indic_3 = \
        higher_bead.get_indicator_functions()

    t_ij = lower_bead_indic_0 @ higher_bead_indic_0 + lower_bead_indic_1 @ higher_bead_indic_1 + \
           lower_bead_indic_2 @ higher_bead_indic_2 + lower_bead_indic_3 @ higher_bead_indic_3
    return t_ij


def _create_h_back(peptide: Peptide, lambda_back):
    main_chain = peptide.get_main_chain
    main_chain_len = len(main_chain)
    full_identity = _build_full_identity(main_chain_len - 1)  # TODO make sure correct
    h_back = full_identity ^ full_identity
    for i in range(len(main_chain) - 2):
        h_back += lambda_back * _check_turns(main_chain[i], main_chain[i + 1])
    h_back -= full_identity ^ full_identity
    h_back = h_back.reduce()
    return h_back


def _create_H_chiral(peptide, lambda_chiral):
    """
    Creates a penalty/constrain term to the total Hamiltonian that imposes that all the position
    of all side chain beads impose the right chirality. Note that the position of the side chain
    bead at a location (i) is determined by the turn indicators at i - 1 and i. In the absence
    of side chains, this function returns a value of 0.

    Args:
        N: Number of total beads in peptide
        side_chain: List of side chains in peptide
        lambda_chiral: Penalty/constraint to impose the right chirality
        indic_0: Turn indicator for axis 0
        indic_1: Turn indicator for axis 1
        indic_2: Turn indicator for axis 2
        indic_3: Turn indicator for axis 3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation

    Returns:
        H_chiral: Hamiltonian term in symbolic notation that imposes the right chirality
    """

    main_chain = peptide.get_main_chain
    main_chain_len = len(main_chain)
    full_identity = _build_full_identity(main_chain_len - 1)
    H_chiral = full_identity ^ full_identity
    for i in range(1, len(main_chain)):  # TODO double check range
        higher_main_bead = main_chain[i]

        if not higher_main_bead.side_chain:
            continue

        higher_side_bead = higher_main_bead.side_chain[0]

        lower_main_bead = main_chain[i - 1]

        lower_main_bead_indic_0, lower_main_bead_indic_1, lower_main_bead_indic_2, \
        lower_main_bead_indic_3 = \
            lower_main_bead.get_indicator_functions()

        higher_main_bead_indic_0, higher_main_bead_indic_1, higher_main_bead_indic_2, \
        higher_main_bead_indic_3 = \
            higher_main_bead.get_indicator_functions()
        higher_side_bead_indic_0, higher_side_bead_indic_1, higher_side_bead_indic_2, \
        higher_side_bead_indic_3 = \
            higher_side_bead.get_indicator_functions()

        si = int((1 - (-1) ** i) / 2)
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_0) @ ((1 - si) * (
                lower_main_bead_indic_1 @ higher_main_bead_indic_2 + lower_main_bead_indic_2 @
                higher_main_bead_indic_3 +
                lower_main_bead_indic_3 @ higher_main_bead_indic_1) + si * (
                                                                              lower_main_bead_indic_2 @ higher_main_bead_indic_1 +
                                                                              lower_main_bead_indic_3 @ higher_main_bead_indic_2 +
                                                                              lower_main_bead_indic_1 @ higher_main_bead_indic_3))
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_1) @ ((1 - si) * (
                lower_main_bead_indic_0 @ higher_main_bead_indic_3 + lower_main_bead_indic_2 @
                higher_main_bead_indic_0 +
                lower_main_bead_indic_3 @ higher_main_bead_indic_2) + si * (
                                                                              lower_main_bead_indic_3 @ higher_main_bead_indic_0 +
                                                                              lower_main_bead_indic_0 @ higher_main_bead_indic_2 +
                                                                              lower_main_bead_indic_2 @ higher_main_bead_indic_3))
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_2) @ ((1 - si) * (
                lower_main_bead_indic_0 @ higher_main_bead_indic_1 + lower_main_bead_indic_1 @
                higher_main_bead_indic_3 +
                lower_main_bead_indic_3 @ higher_main_bead_indic_0) + si * (
                                                                              lower_main_bead_indic_1 @ higher_main_bead_indic_0 +
                                                                              lower_main_bead_indic_3 @ higher_main_bead_indic_1 +
                                                                              lower_main_bead_indic_0 @ higher_main_bead_indic_3))
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_3) @ ((1 - si) * (
                lower_main_bead_indic_0 @ higher_main_bead_indic_2 + lower_main_bead_indic_1 @
                higher_main_bead_indic_0 +
                lower_main_bead_indic_2 @ higher_main_bead_indic_1) + si * (
                                                                              lower_main_bead_indic_2 @ higher_main_bead_indic_0 +
                                                                              lower_main_bead_indic_0 @ higher_main_bead_indic_1 +
                                                                              lower_main_bead_indic_1 @ higher_main_bead_indic_2))
        H_chiral -= full_identity ^ full_identity
        H_chiral = H_chiral.reduce()
    return H_chiral
