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
"""Builds qubit operators for all Hamiltonian terms in the protein folding problem."""
from typing import Union

import numpy as np
from qiskit.opflow import OperatorBase, PauliOp, PauliSumOp

from .bead_contacts.contact_map import ContactMap
from .bead_distances.distance_map import DistanceMap
from .exceptions.invalid_side_chain_exception import (
    InvalidSideChainException,
)
from .exceptions.invalid_size_exception import InvalidSizeException
from .penalty_parameters import PenaltyParameters
from .peptide.pauli_ops_builder import _build_full_identity
from .qubit_utils.qubit_fixing import _fix_qubits
from .peptide.beads.base_bead import BaseBead
from .peptide.peptide import Peptide


# TODO link to a SamplingQubitOpBuilder interface
def _build_qubit_op(
    peptide: Peptide,
    pair_energies: np.ndarray,
    penalty_parameters: PenaltyParameters,
    n_contacts: int,
) -> Union[PauliSumOp, PauliOp]:
    """
    Builds a qubit operator for a total Hamiltonian for a protein folding problem. It includes
    8 terms responsible for chirality, geometry and nearest neighbors interactions.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        pair_energies: Numpy array of pair energies for amino acids.
        penalty_parameters: A PenaltyParameters object storing the values of all penalty
                            parameters.
        n_contacts: number of contacts between all beads.


    Returns:
        h_total: A total Hamiltonian for the protein folding problem.

    Raises:
        InvalidSizeException: if chains of invalid/incompatible sizes provided.
        InvalidSideChainException: if side chains on forbidden indices provided.
    """
    lambda_chiral, lambda_back, lambda_1, lambda_contacts = (
        penalty_parameters.lambda_chiral,
        penalty_parameters.lambda_back,
        penalty_parameters.lambda_1,
        penalty_parameters.lambda_contacts,
    )
    side_chain = peptide.get_side_chain_hot_vector()
    main_chain_len = len(peptide.get_main_chain)

    if len(side_chain) != main_chain_len:
        raise InvalidSizeException("side_chain_lens size not equal main_chain_len")
    if side_chain[0] == 1 or side_chain[-1] == 1 or side_chain[1] == 1:
        raise InvalidSideChainException(
            "First, second and last main beads are not allowed to have a side chain. Non-None "
            "residue provided for an invalid side chain"
        )

    distance_map = DistanceMap(peptide)
    h_chiral = _create_h_chiral(peptide, lambda_chiral)
    h_back = _create_h_back(peptide, lambda_back)

    contact_map = ContactMap(peptide)

    h_scsc = _create_h_scsc(peptide, lambda_1, pair_energies, distance_map, contact_map)
    h_bbbb = _create_h_bbbb(peptide, lambda_1, pair_energies, distance_map, contact_map)

    h_short = _create_h_short(peptide, pair_energies)

    h_bbsc, h_scbb = _create_h_bbsc_and_h_scbb(
        peptide, lambda_1, pair_energies, distance_map, contact_map
    )
    h_contacts = _create_h_contacts(peptide, contact_map, lambda_contacts, n_contacts)

    h_total = h_chiral + h_back + h_short + h_bbbb + h_bbsc + h_scbb + h_scsc + h_contacts

    return h_total.reduce()


def _create_turn_operators(lower_bead: BaseBead, upper_bead: BaseBead) -> OperatorBase:
    """
    Creates a qubit operator for consecutive turns.

    Args:
        lower_bead: A bead with a smaller index in the chain.
        upper_bead: A bead with a bigger index in the chain.

    Returns:
        turns_operator: A qubit operator for consecutive turns.
    """
    (
        lower_bead_indic_0,
        lower_bead_indic_1,
        lower_bead_indic_2,
        lower_bead_indic_3,
    ) = lower_bead.get_indicator_functions()

    (
        upper_bead_indic_0,
        upper_bead_indic_1,
        upper_bead_indic_2,
        upper_bead_indic_3,
    ) = upper_bead.get_indicator_functions()

    turns_operator = _fix_qubits(
        lower_bead_indic_0 @ upper_bead_indic_0
        + lower_bead_indic_1 @ upper_bead_indic_1
        + lower_bead_indic_2 @ upper_bead_indic_2
        + lower_bead_indic_3 @ upper_bead_indic_3
    )
    return turns_operator


def _create_h_back(peptide: Peptide, lambda_back: float) -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian that imposes the geometrical constraint wherein consecutive turns along
    the same axis are penalized by a factor, lambda_back. Note, that the first two turns are
    omitted.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        lambda_back: Constrain that penalizes turns along the same axis.

    Returns:
        H_back: Contribution to Hamiltonian in symbolic notation that penalizes
                consecutive turns along the same axis
    """

    main_chain = peptide.get_main_chain
    h_back = 0
    for i in range(len(main_chain) - 2):
        h_back += lambda_back * _create_turn_operators(main_chain[i], main_chain[i + 1])

    h_back = _fix_qubits(h_back)
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

        (
            lower_main_bead_indic_0,
            lower_main_bead_indic_1,
            lower_main_bead_indic_2,
            lower_main_bead_indic_3,
        ) = lower_main_bead.get_indicator_functions()

        (
            upper_main_bead_indic_0,
            upper_main_bead_indic_1,
            upper_main_bead_indic_2,
            upper_main_bead_indic_3,
        ) = upper_main_bead.get_indicator_functions()
        (
            upper_side_bead_indic_0,
            upper_side_bead_indic_1,
            upper_side_bead_indic_2,
            upper_side_bead_indic_3,
        ) = upper_side_bead.get_indicator_functions()

        turn_coeff = int((1 - (-1) ** i) / 2)
        h_chiral += (
            lambda_chiral
            * (full_id - upper_side_bead_indic_0)
            @ (
                (1 - turn_coeff)
                * (
                    lower_main_bead_indic_1 @ upper_main_bead_indic_2
                    + lower_main_bead_indic_2 @ upper_main_bead_indic_3
                    + lower_main_bead_indic_3 @ upper_main_bead_indic_1
                )
                + turn_coeff
                * (
                    lower_main_bead_indic_2 @ upper_main_bead_indic_1
                    + lower_main_bead_indic_3 @ upper_main_bead_indic_2
                    + lower_main_bead_indic_1 @ upper_main_bead_indic_3
                )
            )
        )
        h_chiral += (
            lambda_chiral
            * (full_id - upper_side_bead_indic_1)
            @ (
                (1 - turn_coeff)
                * (
                    lower_main_bead_indic_0 @ upper_main_bead_indic_3
                    + lower_main_bead_indic_2 @ upper_main_bead_indic_0
                    + lower_main_bead_indic_3 @ upper_main_bead_indic_2
                )
                + turn_coeff
                * (
                    lower_main_bead_indic_3 @ upper_main_bead_indic_0
                    + lower_main_bead_indic_0 @ upper_main_bead_indic_2
                    + lower_main_bead_indic_2 @ upper_main_bead_indic_3
                )
            )
        )
        h_chiral += (
            lambda_chiral
            * (full_id - upper_side_bead_indic_2)
            @ (
                (1 - turn_coeff)
                * (
                    lower_main_bead_indic_0 @ upper_main_bead_indic_1
                    + lower_main_bead_indic_1 @ upper_main_bead_indic_3
                    + lower_main_bead_indic_3 @ upper_main_bead_indic_0
                )
                + turn_coeff
                * (
                    lower_main_bead_indic_1 @ upper_main_bead_indic_0
                    + lower_main_bead_indic_3 @ upper_main_bead_indic_1
                    + lower_main_bead_indic_0 @ upper_main_bead_indic_3
                )
            )
        )
        h_chiral += (
            lambda_chiral
            * (full_id - upper_side_bead_indic_3)
            @ (
                (1 - turn_coeff)
                * (
                    lower_main_bead_indic_0 @ upper_main_bead_indic_2
                    + lower_main_bead_indic_1 @ upper_main_bead_indic_0
                    + lower_main_bead_indic_2 @ upper_main_bead_indic_1
                )
                + turn_coeff
                * (
                    lower_main_bead_indic_2 @ upper_main_bead_indic_0
                    + lower_main_bead_indic_0 @ upper_main_bead_indic_1
                    + lower_main_bead_indic_1 @ upper_main_bead_indic_2
                )
            )
        )
        h_chiral = _fix_qubits(h_chiral)
    return h_chiral


def _create_h_bbbb(
    peptide: Peptide,
    lambda_1: float,
    pair_energies: np.ndarray,
    distance_map: DistanceMap,
    contact_map: ContactMap,
) -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian term corresponding to a 1st neighbor interaction between
    main/backbone (BB) beads.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        distance_map: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        contact_map: ContactMap object that stores contact qubits for all beads.

    Returns:
        h_bbbb: Hamiltonian term corresponding to a 1st neighbor interaction between
                main/backbone (BB) beads.
    """
    h_bbbb = 0
    main_chain_len = len(peptide.get_main_chain)
    for i in range(1, main_chain_len - 3):
        for j in range(i + 5, main_chain_len + 1):
            if (j - i) % 2 == 0:
                continue
            h_bbbb += contact_map.lower_main_upper_main[i][j] @ distance_map._first_neighbor(
                peptide, i, 0, j, 0, lambda_1, pair_energies
            )
            try:
                h_bbbb += contact_map.lower_main_upper_main[i][j] @ distance_map._second_neighbor(
                    peptide, i - 1, 0, j, 0, lambda_1, pair_energies
                )
            except (IndexError, KeyError):
                pass
            try:
                h_bbbb += contact_map.lower_main_upper_main[i][j] @ distance_map._second_neighbor(
                    peptide, i + 1, 0, j, 0, lambda_1, pair_energies
                )
            except (IndexError, KeyError):
                pass
            try:
                h_bbbb += contact_map.lower_main_upper_main[i][j] @ distance_map._second_neighbor(
                    peptide, i, 0, j - 1, 0, lambda_1, pair_energies
                )
            except (IndexError, KeyError):
                pass
            try:
                h_bbbb += contact_map.lower_main_upper_main[i][j] @ distance_map._second_neighbor(
                    peptide, i, 0, j + 1, 0, lambda_1, pair_energies
                )
            except (IndexError, KeyError):
                pass
            h_bbbb = _fix_qubits(h_bbbb)
    return h_bbbb


def _create_h_bbsc_and_h_scbb(
    peptide: Peptide,
    lambda_1: float,
    pair_energies: np.ndarray,
    distance_map: DistanceMap,
    contact_map: ContactMap,
) -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    main/backbone (BB) and side chain (SC) beads. In the absence
    of side chains, this function returns a value of 0.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        distance_map: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        contact_map: ContactMap object that stores contact qubits for all beads.

    Returns:
        h_bbsc, h_scbb: Tuple of Hamiltonian terms consisting of backbone and side chain
        interactions.
    """
    h_bbsc = 0
    h_scbb = 0
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    for i in range(1, main_chain_len - 3):
        for j in range(i + 4, main_chain_len + 1):
            if (j - i) % 2 == 1:
                continue

            if side_chain[j - 1] == 1:

                h_bbsc += contact_map.lower_side_upper_main[i][j] @ (
                    distance_map._first_neighbor(peptide, i, 0, j, 1, lambda_1, pair_energies)
                    + distance_map._second_neighbor(peptide, i, 0, j, 0, lambda_1, pair_energies)
                )
                try:
                    h_bbsc += contact_map.lower_side_upper_side[i][
                        j
                    ] @ distance_map._first_neighbor(peptide, i, 1, j, 1, lambda_1, pair_energies)
                except (IndexError, KeyError, TypeError):
                    pass
                try:
                    h_bbsc += contact_map.lower_side_upper_main[i][
                        j
                    ] @ distance_map._second_neighbor(
                        peptide, i + 1, 0, j, 1, lambda_1, pair_energies
                    )
                except (IndexError, KeyError, TypeError):
                    pass
                try:
                    h_bbsc += contact_map.lower_side_upper_main[i][
                        j
                    ] @ distance_map._second_neighbor(
                        peptide, i - 1, 0, j, 1, lambda_1, pair_energies
                    )
                except (IndexError, KeyError, TypeError):
                    pass
            if side_chain[i - 1] == 1:
                h_scbb += contact_map.lower_main_upper_side[i][j] @ (
                    distance_map._first_neighbor(peptide, i, 1, j, 0, lambda_1, pair_energies)
                    + distance_map._second_neighbor(peptide, i, 0, j, 0, lambda_1, pair_energies)
                )
                try:
                    h_scbb += contact_map.lower_main_upper_side[i][
                        j
                    ] @ distance_map._second_neighbor(peptide, i, 1, j, 1, lambda_1, pair_energies)
                except (IndexError, KeyError, TypeError):
                    pass
                try:
                    h_scbb += contact_map.lower_main_upper_side[i][
                        j
                    ] @ distance_map._second_neighbor(
                        peptide, i, 1, j + 1, 0, lambda_1, pair_energies
                    )
                except (IndexError, KeyError, TypeError):
                    pass
                try:
                    h_scbb += contact_map.lower_main_upper_side[i][
                        j
                    ] @ distance_map._second_neighbor(
                        peptide, i, 1, j - 1, 0, lambda_1, pair_energies
                    )
                except (IndexError, KeyError, TypeError):
                    pass

    h_bbsc = _fix_qubits(h_bbsc)
    h_scbb = _fix_qubits(h_scbb)
    return h_bbsc, h_scbb


def _create_h_scsc(
    peptide: Peptide,
    lambda_1: float,
    pair_energies: np.ndarray,
    distance_map: DistanceMap,
    contact_map: ContactMap,
) -> Union[PauliSumOp, PauliOp]:
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    side chain (SC) beads. In the absence of side chains, this function
    returns a value of 0.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        distance_map: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        contact_map: ContactMap object that stores contact qubits for all beads.

    Returns:
        h_scsc: Hamiltonian term consisting of side chain pairwise interactions
    """
    h_scsc = 0
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    for i in range(1, main_chain_len - 3):
        for j in range(i + 5, main_chain_len + 1):
            if (j - i) % 2 == 0:
                continue
            if side_chain[i - 1] == 0 or side_chain[j - 1] == 0:
                continue
            h_scsc += contact_map.lower_side_upper_side[i][j] @ (
                distance_map._first_neighbor(peptide, i, 1, j, 1, lambda_1, pair_energies)
                + distance_map._second_neighbor(peptide, i, 1, j, 0, lambda_1, pair_energies)
                + distance_map._second_neighbor(peptide, i, 0, j, 1, lambda_1, pair_energies)
            )
    return _fix_qubits(h_scsc)


def _create_h_short(peptide: Peptide, pair_energies: np.ndarray) -> Union[PauliSumOp, PauliOp]:
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
            op1 = _create_turn_operators(
                peptide.get_main_chain[i + 1], peptide.get_main_chain[i - 1].side_chain[0]
            )
            op2 = _create_turn_operators(
                peptide.get_main_chain[i - 1], peptide.get_main_chain[i + 2].side_chain[0]
            )
            coeff = float(
                pair_energies[i][1][i + 3][1]
                + 0.1 * (pair_energies[i][1][i + 3][0] + pair_energies[i][0][i + 3][1])
            )
            composed = op1 @ op2
            h_short += (coeff * composed).reduce()
    h_short = _fix_qubits(h_short)

    return h_short


# TODO in the original code, n_contacts is always set to 0. What is the meaning of this param?
def _create_h_contacts(
    peptide: Peptide, contact_map: ContactMap, lambda_contacts: float, n_contacts: int = 0
) -> Union[PauliSumOp, PauliOp]:
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
    new_qubits = contact_map._create_peptide_qubit_list()
    main_chain_len = len(peptide.get_main_chain)
    full_id = _build_full_identity(2 * (main_chain_len - 1))
    # original code treats the 0th entry (valued 0) as a qubit register
    new_qubits[0] = 0.5 * (full_id ^ full_id)
    h_contacts = 0.0
    for operator in new_qubits[-contact_map.num_contacts :]:
        h_contacts += operator
    h_contacts -= n_contacts * (full_id ^ full_id)
    h_contacts = h_contacts ** 2
    h_contacts *= lambda_contacts
    h_contacts = _fix_qubits(h_contacts)
    return h_contacts
