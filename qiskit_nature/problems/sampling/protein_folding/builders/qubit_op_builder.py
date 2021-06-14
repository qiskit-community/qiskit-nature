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
from typing import List, Union

from qiskit.opflow import OperatorBase, PauliOp, PauliSumOp

from problems.sampling.protein_folding.contact_map import ContactMap
from problems.sampling.protein_folding.distance_calculator import _calc_total_distances, \
    _first_neighbor, _second_neighbor
from problems.sampling.protein_folding.exceptions.invalid_side_chain_exception import \
    InvalidSideChainException
from problems.sampling.protein_folding.exceptions.invalid_size_exception import InvalidSizeException
from problems.sampling.protein_folding.penalty_parameters import PenaltyParameters
from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from problems.sampling.protein_folding.qubit_fixing import _fix_qubits
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


def _build_qubit_op(peptide: Peptide, pair_energies: List[List[List[List[float]]]],
                    penalty_parameters: PenaltyParameters, n_contacts) -> Union[
    PauliSumOp, PauliOp]:
    """
        Builds a qubit operator for a total Hamiltonian for a protein folding problem. It includes
        8 terms responsible for chirality, geometry and nearest neighbours interactions.

        Args:
            peptide: A Peptide object that includes all information about a protein.
            pair_energies: Numpy array of pair energies for amino acids.
            penalty_parameters: A PenaltyParameters object storing the values of all penalty
                                parameters.
            n_contacts: # TODO


        Returns:
            h_total: A total Hamiltonian for the protein folding problem.
        """
    lambda_chiral, lambda_back, lambda_1, lambda_contacts = penalty_parameters.lambda_chiral, \
                                                            penalty_parameters.lambda_back, \
                                                            penalty_parameters.lambda_1, \
                                                            penalty_parameters.lambda_contacts
    side_chain = peptide.get_side_chain_hot_vector()
    main_chain_len = len(peptide.get_main_chain)

    if len(side_chain) != main_chain_len:
        raise InvalidSizeException("side_chain_lens size not equal main_chain_len")
    if side_chain[0] == 1 or side_chain[-1] == 1 or side_chain[1] == 1:
        raise InvalidSideChainException(
            "First, second and last main beads are not allowed to have a side chain. Non-None "
            "residue provided for an invalid side chain")

    x_dist = _calc_total_distances(peptide)
    h_chiral = _create_h_chiral(peptide, lambda_chiral)
    h_back = _create_h_back(peptide, lambda_back)

    contact_map = ContactMap(peptide)

    h_scsc = _create_h_scsc(main_chain_len, side_chain, lambda_1,
                            pair_energies, x_dist, contact_map)
    h_bbbb = _create_h_bbbb(main_chain_len, lambda_1, pair_energies, x_dist, contact_map)

    h_short = _create_h_short(peptide, pair_energies)

    h_bbsc, h_scbb = _create_h_bbsc_and_h_scbb(main_chain_len, side_chain, lambda_1,
                                               pair_energies, x_dist, contact_map)
    h_contacts = _create_h_contacts(peptide, contact_map, lambda_contacts, n_contacts)

    h_tot = h_chiral + h_back + h_short + h_bbbb + h_bbsc + h_scbb + h_scsc + h_contacts

    return h_tot.reduce()


def _check_turns(lower_bead: BaseBead, upper_bead: BaseBead) -> OperatorBase:
    lower_bead_indic_0, lower_bead_indic_1, lower_bead_indic_2, lower_bead_indic_3 = \
        lower_bead.get_indicator_functions()

    upper_bead_indic_0, upper_bead_indic_1, upper_bead_indic_2, upper_bead_indic_3 = \
        upper_bead.get_indicator_functions()

    t_ij = _fix_qubits(
        lower_bead_indic_0 @ upper_bead_indic_0 + lower_bead_indic_1 @ upper_bead_indic_1 + \
        lower_bead_indic_2 @ upper_bead_indic_2 + lower_bead_indic_3 @ upper_bead_indic_3)
    return t_ij


def _create_h_back(peptide: Peptide, lambda_back: float) -> Union[PauliSumOp, PauliOp]:
    main_chain = peptide.get_main_chain
    h_back = 0
    for i in range(len(main_chain) - 2):
        h_back += lambda_back * _check_turns(main_chain[i], main_chain[i + 1])

    h_back = _fix_qubits(h_back).reduce()
    return h_back


def _create_h_chiral(peptide: Peptide, lambda_chiral: float) -> Union[PauliSumOp, PauliOp]:
    """
    Creates a penalty/constrain term to the total Hamiltonian that imposes that all the position
    of all side chain beads impose the right chirality. Note that the position of the side chain
    bead at a location (i) is determined by the turn indicators at i - 1 and i. In the absence
    of side chains, this function returns a value of 0.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        lambda_chiral: Penalty/constraint to impose the right chirality.

    Returns:
        h_chiral: Hamiltonian term that imposes the right chirality.
    """

    main_chain = peptide.get_main_chain
    main_chain_len = len(main_chain)
    # 2 stands for 2 qubits per turn, another 2 stands for main and side qubit register
    h_chiral = 0
    full_id = _build_full_identity(2 * 2 * (main_chain_len - 1))
    for i in range(1, len(main_chain) + 1):
        upper_main_bead = main_chain[i - 1]

        if upper_main_bead.side_chain is None:
            continue

        upper_side_bead = upper_main_bead.side_chain[0]

        lower_main_bead = main_chain[i - 2]

        lower_main_bead_indic_0, lower_main_bead_indic_1, lower_main_bead_indic_2, \
        lower_main_bead_indic_3 = \
            lower_main_bead.get_indicator_functions()

        upper_main_bead_indic_0, upper_main_bead_indic_1, upper_main_bead_indic_2, \
        upper_main_bead_indic_3 = \
            upper_main_bead.get_indicator_functions()
        upper_side_bead_indic_0, upper_side_bead_indic_1, upper_side_bead_indic_2, \
        upper_side_bead_indic_3 = \
            upper_side_bead.get_indicator_functions()

        si = int((1 - (-1) ** i) / 2)
        h_chiral += lambda_chiral * (full_id - upper_side_bead_indic_0) @ ((1 - si) * (
                lower_main_bead_indic_1 @ upper_main_bead_indic_2 + lower_main_bead_indic_2 @
                upper_main_bead_indic_3 +
                lower_main_bead_indic_3 @ upper_main_bead_indic_1) + si * (
                                                                                   lower_main_bead_indic_2 @ upper_main_bead_indic_1 +
                                                                                   lower_main_bead_indic_3 @ upper_main_bead_indic_2 +
                                                                                   lower_main_bead_indic_1 @ upper_main_bead_indic_3))
        h_chiral += lambda_chiral * (full_id - upper_side_bead_indic_1) @ ((1 - si) * (
                lower_main_bead_indic_0 @ upper_main_bead_indic_3 + lower_main_bead_indic_2 @
                upper_main_bead_indic_0 +
                lower_main_bead_indic_3 @ upper_main_bead_indic_2) + si * (
                                                                                   lower_main_bead_indic_3 @ upper_main_bead_indic_0 +
                                                                                   lower_main_bead_indic_0 @ upper_main_bead_indic_2 +
                                                                                   lower_main_bead_indic_2 @ upper_main_bead_indic_3))
        h_chiral += lambda_chiral * (full_id - upper_side_bead_indic_2) @ ((1 - si) * (
                lower_main_bead_indic_0 @ upper_main_bead_indic_1 + lower_main_bead_indic_1 @
                upper_main_bead_indic_3 +
                lower_main_bead_indic_3 @ upper_main_bead_indic_0) + si * (
                                                                                   lower_main_bead_indic_1 @ upper_main_bead_indic_0 +
                                                                                   lower_main_bead_indic_3 @ upper_main_bead_indic_1 +
                                                                                   lower_main_bead_indic_0 @ upper_main_bead_indic_3))
        h_chiral += lambda_chiral * (full_id - upper_side_bead_indic_3) @ ((1 - si) * (
                lower_main_bead_indic_0 @ upper_main_bead_indic_2 + lower_main_bead_indic_1 @
                upper_main_bead_indic_0 +
                lower_main_bead_indic_2 @ upper_main_bead_indic_1) + si * (
                                                                                   lower_main_bead_indic_2 @ upper_main_bead_indic_0 +
                                                                                   lower_main_bead_indic_0 @ upper_main_bead_indic_1 +
                                                                                   lower_main_bead_indic_1 @ upper_main_bead_indic_2))
        h_chiral = _fix_qubits(h_chiral).reduce()
    return h_chiral


def _create_h_bbbb(main_chain_len: int, lambda_1: float,
                   pair_energies: List[List[List[List[float]]]],
                   x_dist, contact_map: ContactMap) -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian term corresponding to a 1st neighbor interaction between
    main/backbone (BB) beads.

    Args:
        main_chain_len: Number of total beads in a peptide.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        contact_map: ContactMap object that stores contact qubits for all beads.

    Returns:
        H_BBBB: Hamiltonian term corresponding to a 1st neighbor interaction between
                main/backbone (BB) beads.
    """
    H_BBBB = 0
    for i in range(1, main_chain_len - 3):
        for j in range(i + 5, main_chain_len + 1):
            if (j - i) % 2 == 0:
                continue
            else:
                H_BBBB += contact_map.lower_main_upper_main[i][j] @ _first_neighbor(i, 0, j, 0,
                                                                                    lambda_1,
                                                                                    pair_energies,
                                                                                    x_dist)
                try:
                    H_BBBB += contact_map.lower_main_upper_main[i][j] @ _second_neighbor(i - 1, 0,
                                                                                         j, 0,
                                                                                         lambda_1,
                                                                                         pair_energies,
                                                                                         x_dist)
                except (IndexError, KeyError):
                    pass
                try:
                    H_BBBB += contact_map.lower_main_upper_main[i][j] @ _second_neighbor(i + 1, 0,
                                                                                         j, 0,
                                                                                         lambda_1,
                                                                                         pair_energies,
                                                                                         x_dist)
                except (IndexError, KeyError):
                    pass
                try:
                    H_BBBB += contact_map.lower_main_upper_main[i][j] @ _second_neighbor(i, 0,
                                                                                         j - 1, 0,
                                                                                         lambda_1,
                                                                                         pair_energies,
                                                                                         x_dist)
                except (IndexError, KeyError):
                    pass
                try:
                    H_BBBB += contact_map.lower_main_upper_main[i][j] @ _second_neighbor(i, 0,
                                                                                         j + 1, 0,
                                                                                         lambda_1,
                                                                                         pair_energies,
                                                                                         x_dist)
                except (IndexError, KeyError):
                    pass
            H_BBBB = _fix_qubits(H_BBBB).reduce()
    return H_BBBB


def _create_h_bbsc_and_h_scbb(main_chain_len: int, side_chain: List[int], lambda_1: float,
                              pair_energies: List[List[List[List[float]]]], x_dist,
                              contact_map: ContactMap) -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    main/backbone (BB) and side chain (SC) beads. In the absence
    of side chains, this function returns a value of 0.

    Args:
        main_chain_len: Number of total beads in peptide
        side_chain: One-hot encoded list of side chains in peptide.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        contact_map: ContactMap object that stores contact qubits for all beads.

    Returns:
        h_bbsc, h_scbb: Tuple of Hamiltonian terms consisting of backbone and side chain
        interactions.
    """
    h_bbsc = 0
    h_scbb = 0
    for i in range(1, main_chain_len - 3):
        for j in range(i + 4, main_chain_len + 1):
            if (j - i) % 2 == 1:
                continue
            else:
                if side_chain[j - 1] == 1:
                    h_bbsc += (contact_map.lower_side_upper_main[i][j] @ (
                            _first_neighbor(i, 0, j, 1, lambda_1, pair_energies, x_dist) +
                            _second_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist)))
                    try:
                        h_bbsc += (contact_map.lower_side_upper_main[i][j] @ _first_neighbor(i, 1,
                                                                                             j, 1,
                                                                                             lambda_1,
                                                                                             pair_energies,
                                                                                             x_dist))
                    except (IndexError, KeyError):
                        pass
                    try:
                        h_bbsc += (contact_map.lower_side_upper_main[i][j] @ _second_neighbor(i + 1,
                                                                                              0, j,
                                                                                              1,
                                                                                              lambda_1,
                                                                                              pair_energies,
                                                                                              x_dist))
                    except (IndexError, KeyError):
                        pass
                    try:
                        h_bbsc += (contact_map.lower_side_upper_main[i][j] @ _second_neighbor(i - 1,
                                                                                              0, j,
                                                                                              1,
                                                                                              lambda_1,
                                                                                              pair_energies,
                                                                                              x_dist))
                    except (IndexError, KeyError):
                        pass
                    h_bbsc = h_bbsc.reduce()
                if side_chain[i - 1] == 1:
                    h_scbb += (contact_map.lower_main_upper_side[i][j] @ (
                            _first_neighbor(i, 1, j, 0, lambda_1, pair_energies, x_dist) +
                            _second_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist)))
                    try:
                        h_scbb += (contact_map.lower_main_upper_side[i][j] @ _second_neighbor(i, 1,
                                                                                              j, 1,
                                                                                              lambda_1,
                                                                                              pair_energies,
                                                                                              x_dist))
                    except (IndexError, KeyError):
                        pass
                    try:
                        h_scbb += (contact_map.lower_main_upper_side[i][j] @ _second_neighbor(i, 1,
                                                                                              j + 1,
                                                                                              0,
                                                                                              lambda_1,
                                                                                              pair_energies,
                                                                                              x_dist))
                    except (IndexError, KeyError):
                        pass
                    try:
                        h_scbb += (contact_map.lower_main_upper_side[i][j] @ _second_neighbor(i, 1,
                                                                                              j - 1,
                                                                                              0,
                                                                                              lambda_1,
                                                                                              pair_energies,
                                                                                              x_dist))
                    except (IndexError, KeyError):
                        pass
                    h_scbb = h_scbb.reduce()

    if h_bbsc != 0 and h_bbsc is not None:
        h_bbsc = _fix_qubits(h_bbsc).reduce()
    if h_scbb != 0 and h_scbb is not None:
        h_scbb = _fix_qubits(h_scbb).reduce()
    return h_bbsc, h_scbb


def _create_h_scsc(main_chain_len: int, side_chain: List[int], lambda_1: float,
                   pair_energies: List[List[List[List[float]]]], x_dist, contact_map: ContactMap) \
        -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    side chain (SC) beads. In the absence of side chains, this function
    returns a value of 0.

    Args:
        main_chain_len: Number of total beads in peptides.
        side_chain: One-hot encoded list of side chains in peptide.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        contact_map: ContactMap object that stores contact qubits for all beads.

    Returns:
        h_scsc: Hamiltonian term consisting of side chain pairwise interactions
    """
    h_scsc = 0
    for i in range(1, main_chain_len - 3):
        for j in range(i + 5, main_chain_len + 1):
            if (j - i) % 2 == 0:
                continue
            if side_chain[i - 1] == 0 or side_chain[j - 1] == 0:
                continue
            h_scsc += contact_map.lower_side_upper_side[i][j] @ (
                    _first_neighbor(i, 1, j, 1, lambda_1, pair_energies, x_dist) +
                    _second_neighbor(i, 1, j, 0, lambda_1, pair_energies, x_dist)
                    +
                    _second_neighbor(i, 0, j, 1, lambda_1, pair_energies, x_dist))
            h_scsc = h_scsc.reduce()
    return _fix_qubits(h_scsc).reduce()


def _create_h_short(peptide: Peptide, pair_energies: List[List[List[List[float]]]]) -> Union[
    PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian constituting interactions between beads that are no more than
    4 beads apart. If no side chains are present, this function returns 0.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        pair_energies: Numpy array of pair energies for amino acids.

    Returns:
        h_short: Contribution to energetic Hamiltonian for interactions between beads that are no
        more than 4 beads apart.
    """
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    h_short = 0
    for i in range(1, main_chain_len - 2):
        # checks interactions between beads no more than 4 beads apart
        if side_chain[i - 1] == 1 and side_chain[i + 2] == 1:
            op1 = _check_turns(peptide.get_main_chain[i + 1],
                               peptide.get_main_chain[i - 1].side_chain[0])
            op2 = _check_turns(peptide.get_main_chain[i - 1],
                               peptide.get_main_chain[i + 2].side_chain[0])
            coeff = float(pair_energies[i, 1, i + 3, 1] + 0.1 * (
                    pair_energies[i, 1, i + 3, 0] + pair_energies[i, 0, i + 3, 1]))
            composed = op1 @ op2
            h_short += (coeff * composed).reduce()
    h_short = _fix_qubits(h_short).reduce()

    return h_short


# TODO in the original code, n_contacts is always set to 0. What is the meaning of this param?
def _create_h_contacts(peptide: Peptide, contact_map: ContactMap, lambda_contacts: float,
                       n_contacts: int = 0) -> Union[PauliSumOp, PauliOp]:
    """
    Creates a Hamiltonian term approximating nearest neighbor interactions and includes energy of
    contacts that are present in system (energy shift). # TODO better description?

    Args:
        peptide: A Peptide object that includes all information about a protein.
        contact_map: ContactMap object that stores contact qubits for all beads.
        lambda_contacts: Constraint to penalize local overlap between beads within a nearest
        neighbor contact.
        n_contacts: # TODO
    Returns:
        h_contacts: Contribution to energetic Hamiltonian for approximate nearest neighbor
        interactions.

    """
    new_qubits = contact_map.create_peptide_qubit_list()
    main_chain_len = len(peptide.get_main_chain)
    full_id = _build_full_identity(2 * (main_chain_len - 1))
    h_contacts = 0
    for el in new_qubits[-contact_map.r_contact:]:
        h_contacts += el
    h_contacts -= n_contacts * (full_id ^ full_id)
    h_contacts = h_contacts ** 2
    h_contacts *= lambda_contacts
    h_contacts = _fix_qubits(h_contacts).reduce()
    return h_contacts
