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


class QubitOpBuilder:
    """Builds qubit operators for all Hamiltonian terms in the protein folding problem."""

    def __init__(
            self, peptide: Peptide, pair_energies: np.ndarray, penalty_parameters: PenaltyParameters
    ):
        """Builds qubit operators for all Hamiltonian terms in the protein folding problem.

        Args:
            peptide: A Peptide object that includes all information about a protein.
            pair_energies: Numpy array of pair energies for amino acids.
            penalty_parameters: A PenaltyParameters object storing the values of all penalty
                                parameters.
        """
        self._peptide = peptide
        self._pair_energies = pair_energies
        self._penalty_parameters = penalty_parameters
        self._contact_map = ContactMap(peptide)
        self._distance_map = DistanceMap(peptide)
        _side_chain_hot_vector = self._peptide.get_side_chain_hot_vector()
        self._has_side_chain_second_bead = _side_chain_hot_vector[1] if len(
            _side_chain_hot_vector) > 1 else False

    def _build_qubit_op(self) -> Union[PauliSumOp, PauliOp]:
        """
        Builds a qubit operator for a total Hamiltonian for a protein folding problem. It includes
        8 terms responsible for chirality, geometry and nearest neighbors interactions.

        Returns:
            A total Hamiltonian for the protein folding problem.

        Raises:
            InvalidSizeException: if chains of invalid/incompatible sizes provided.
            InvalidSideChainException: if side chains on forbidden indices provided.
        """
        side_chain = self._peptide.get_side_chain_hot_vector()
        main_chain_len = len(self._peptide.get_main_chain)

        if len(side_chain) != main_chain_len:
            raise InvalidSizeException("side_chain_lens size not equal main_chain_len")
        if side_chain[0] == 1 or side_chain[-1] == 1 or side_chain[1] == 1:
            raise InvalidSideChainException(
                "First, second and last main beads are not allowed to have a side chain. Non-None "
                "residue provided for an invalid side chain"
            )

        num_qubits = 4 * pow(main_chain_len - 1, 2)
        full_id = _build_full_identity(num_qubits)

        h_chiral = self._create_h_chiral()
        if h_chiral != 0:
            h_chiral = full_id ^ h_chiral
        h_back = self._create_h_back()
        if h_back != 0:
            h_back = full_id ^ h_back

        h_scsc = self._create_h_scsc() if self._penalty_parameters.penalty_1 else 0
        h_bbbb = self._create_h_bbbb() if self._penalty_parameters.penalty_1 else 0

        h_short = self._create_h_short()
        if h_short != 0:
            h_short = full_id ^ h_short

        h_bbsc, h_scbb = (
            self._create_h_bbsc_and_h_scbb() if self._penalty_parameters.penalty_1 else (0, 0)
        )

        h_total = h_chiral + h_back + h_short + h_bbbb + h_bbsc + h_scbb + h_scsc

        return h_total.reduce()

    def _create_turn_operators(self, lower_bead: BaseBead, upper_bead: BaseBead) -> OperatorBase:
        """
        Creates a qubit operator for consecutive turns.

        Args:
            lower_bead: A bead with a smaller index in the chain.
            upper_bead: A bead with a bigger index in the chain.

        Returns:
            A qubit operator for consecutive turns.
        """
        (
            lower_bead_indic_0,
            lower_bead_indic_1,
            lower_bead_indic_2,
            lower_bead_indic_3,
        ) = lower_bead.indicator_functions

        (
            upper_bead_indic_0,
            upper_bead_indic_1,
            upper_bead_indic_2,
            upper_bead_indic_3,
        ) = upper_bead.indicator_functions

        turns_operator = _fix_qubits(
            lower_bead_indic_0 @ upper_bead_indic_0
            + lower_bead_indic_1 @ upper_bead_indic_1
            + lower_bead_indic_2 @ upper_bead_indic_2
            + lower_bead_indic_3 @ upper_bead_indic_3 ,self._has_side_chain_second_bead)
        return turns_operator

    def _create_h_back(self) -> Union[PauliSumOp, PauliOp]:
        """
        Creates Hamiltonian that imposes the geometrical constraint wherein consecutive turns along
        the same axis are penalized by a factor, penalty_back. Note, that the first two turns are
        omitted (fixed in optimization) due to symmetry degeneracy.

        Returns:
            Contribution to Hamiltonian in symbolic notation that penalizes consecutive turns
            along the same axis.
        """

        main_chain = self._peptide.get_main_chain
        penalty_back = self._penalty_parameters.penalty_back
        h_back = 0
        for i in range(len(main_chain) - 2):
            h_back += penalty_back * self._create_turn_operators(main_chain[i], main_chain[i + 1])

        h_back = _fix_qubits(h_back, self._has_side_chain_second_bead)
        return h_back

    def _create_h_chiral(self) -> Union[PauliSumOp, PauliOp]:
        """
        Creates a penalty/constrain term to the total Hamiltonian that imposes that all the position
        of all side chain beads impose the right chirality. Note that the position of the side chain
        bead at a location (i) is determined by the turn indicators at i - 1 and i. In the absence
        of side chains, this function returns a value of 0.

        Returns:
            Hamiltonian term that imposes the right chirality.
        """

        main_chain = self._peptide.get_main_chain
        main_chain_len = len(main_chain)
        h_chiral = 0
        # 2 stands for 2 qubits per turn, another 2 stands for main and side qubit register
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
            ) = lower_main_bead.indicator_functions

            (
                upper_main_bead_indic_0,
                upper_main_bead_indic_1,
                upper_main_bead_indic_2,
                upper_main_bead_indic_3,
            ) = upper_main_bead.indicator_functions
            (
                upper_side_bead_indic_0,
                upper_side_bead_indic_1,
                upper_side_bead_indic_2,
                upper_side_bead_indic_3,
            ) = upper_side_bead.indicator_functions

            turn_coeff = int((1 - (-1) ** i) / 2)
            h_chiral += self._build_chiral_term(
                full_id,
                lower_main_bead_indic_1,
                lower_main_bead_indic_2,
                lower_main_bead_indic_3,
                turn_coeff,
                upper_main_bead_indic_1,
                upper_main_bead_indic_2,
                upper_main_bead_indic_3,
                upper_side_bead_indic_0,
            )
            h_chiral += self._build_chiral_term(
                full_id,
                lower_main_bead_indic_0,
                lower_main_bead_indic_3,
                lower_main_bead_indic_2,
                turn_coeff,
                upper_main_bead_indic_0,
                upper_main_bead_indic_3,
                upper_main_bead_indic_2,
                upper_side_bead_indic_1,
            )
            h_chiral += self._build_chiral_term(
                full_id,
                lower_main_bead_indic_0,
                lower_main_bead_indic_1,
                lower_main_bead_indic_3,
                turn_coeff,
                upper_main_bead_indic_0,
                upper_main_bead_indic_1,
                upper_main_bead_indic_3,
                upper_side_bead_indic_2,
            )
            h_chiral += self._build_chiral_term(
                full_id,
                lower_main_bead_indic_0,
                lower_main_bead_indic_2,
                lower_main_bead_indic_1,
                turn_coeff,
                upper_main_bead_indic_0,
                upper_main_bead_indic_2,
                upper_main_bead_indic_1,
                upper_side_bead_indic_3,
            )
            h_chiral = _fix_qubits(h_chiral, self._has_side_chain_second_bead)
        return h_chiral

    def _build_chiral_term(
            self,
            full_id,
            lower_main_bead_indic_b,
            lower_main_bead_indic_c,
            lower_main_bead_indic_d,
            turn_coeff,
            upper_main_bead_indic_b,
            upper_main_bead_indic_c,
            upper_main_bead_indic_d,
            upper_side_bead_indic_a,
    ):
        return (
                self._penalty_parameters.penalty_chiral
                * (full_id - upper_side_bead_indic_a)
                @ (
                        (1 - turn_coeff)
                        * (
                                lower_main_bead_indic_b @ upper_main_bead_indic_c
                                + lower_main_bead_indic_c @ upper_main_bead_indic_d
                                + lower_main_bead_indic_d @ upper_main_bead_indic_b
                        )
                        + turn_coeff
                        * (
                                lower_main_bead_indic_c @ upper_main_bead_indic_b
                                + lower_main_bead_indic_d @ upper_main_bead_indic_c
                                + lower_main_bead_indic_b @ upper_main_bead_indic_d
                        )
                )
        )

    def _create_h_bbbb(self) -> Union[PauliSumOp, PauliOp]:
        """
        Creates Hamiltonian term corresponding to a 1st neighbor interaction between
        main/backbone (BB) beads.

        Returns:
            Hamiltonian term corresponding to a 1st neighbor interaction between main/backbone (
            BB) beads.
        """
        penalty_1 = self._penalty_parameters.penalty_1
        h_bbbb = 0
        main_chain_len = len(self._peptide.get_main_chain)
        for i in range(1, main_chain_len - 3):
            for j in range(i + 5, main_chain_len + 1):
                if (j - i) % 2 == 0:
                    continue
                h_bbbb += (self._contact_map.lower_main_upper_main[i][j]) ^ (
                    self._distance_map._first_neighbor(
                        self._peptide, i, 0, j, 0, penalty_1, self._pair_energies
                    )
                )
                try:
                    h_bbbb += (self._contact_map.lower_main_upper_main[i][j]) ^ (
                        self._distance_map._second_neighbor(
                            self._peptide, i - 1, 0, j, 0, penalty_1, self._pair_energies
                        )
                    )
                except (IndexError, KeyError):
                    pass
                try:
                    h_bbbb += (self._contact_map.lower_main_upper_main[i][j]) ^ (
                        self._distance_map._second_neighbor(
                            self._peptide, i + 1, 0, j, 0, penalty_1, self._pair_energies
                        )
                    )
                except (IndexError, KeyError):
                    pass
                try:
                    h_bbbb += (self._contact_map.lower_main_upper_main[i][j]) ^ (
                        self._distance_map._second_neighbor(
                            self._peptide, i, 0, j - 1, 0, penalty_1, self._pair_energies
                        )
                    )
                except (IndexError, KeyError):
                    pass
                try:
                    h_bbbb += (self._contact_map.lower_main_upper_main[i][j]) ^ (
                        self._distance_map._second_neighbor(
                            self._peptide, i, 0, j + 1, 0, penalty_1, self._pair_energies
                        )
                    )
                except (IndexError, KeyError):
                    pass
                h_bbbb = _fix_qubits(h_bbbb, self._has_side_chain_second_bead)
        return h_bbbb

    def _create_h_bbsc_and_h_scbb(self) -> Union[PauliSumOp, PauliOp]:
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        main/backbone (BB) and side chain (SC) beads. In the absence
        of side chains, this function returns a value of 0.

        Returns:
            Tuple of Hamiltonian terms consisting of backbone and side chain interactions.
        """
        penalty_1 = self._penalty_parameters.penalty_1
        h_bbsc = 0
        h_scbb = 0
        main_chain_len = len(self._peptide.get_main_chain)
        side_chain = self._peptide.get_side_chain_hot_vector()
        for i in range(1, main_chain_len - 3):
            for j in range(i + 4, main_chain_len + 1):
                if (j - i) % 2 == 1:
                    continue

                if side_chain[j - 1] == 1:

                    h_bbsc += self._contact_map.lower_main_upper_side[i][j] ^ (
                            self._distance_map._first_neighbor(
                                self._peptide, i, 0, j, 1, penalty_1, self._pair_energies
                            )
                            + self._distance_map._second_neighbor(
                        self._peptide, i, 0, j, 0, penalty_1, self._pair_energies
                    )
                    )
                    try:
                        h_bbsc += self._contact_map.lower_side_upper_side[i][
                                      j
                                  ] ^ self._distance_map._first_neighbor(
                            self._peptide, i, 1, j, 1, penalty_1, self._pair_energies
                        )
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        h_bbsc += self._contact_map.lower_main_upper_side[i][
                                      j
                                  ] ^ self._distance_map._second_neighbor(
                            self._peptide, i + 1, 0, j, 1, penalty_1, self._pair_energies
                        )
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        h_bbsc += self._contact_map.lower_main_upper_side[i][
                                      j
                                  ] ^ self._distance_map._second_neighbor(
                            self._peptide, i - 1, 0, j, 1, penalty_1, self._pair_energies
                        )
                    except (IndexError, KeyError, TypeError):
                        pass
                if side_chain[i - 1] == 1:
                    h_scbb += self._contact_map.lower_side_upper_main[i][j] ^ (
                            self._distance_map._first_neighbor(
                                self._peptide, i, 1, j, 0, penalty_1, self._pair_energies
                            )
                            + self._distance_map._second_neighbor(
                        self._peptide, i, 0, j, 0, penalty_1, self._pair_energies
                    )
                    )
                    try:
                        h_scbb += self._contact_map.lower_side_upper_main[i][
                                      j
                                  ] ^ self._distance_map._second_neighbor(
                            self._peptide, i, 1, j, 1, penalty_1, self._pair_energies
                        )
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        h_scbb += self._contact_map.lower_side_upper_main[i][
                                      j
                                  ] ^ self._distance_map._second_neighbor(
                            self._peptide, i, 1, j + 1, 0, penalty_1, self._pair_energies
                        )
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        h_scbb += self._contact_map.lower_side_upper_main[i][
                                      j
                                  ] ^ self._distance_map._second_neighbor(
                            self._peptide, i, 1, j - 1, 0, penalty_1, self._pair_energies
                        )
                    except (IndexError, KeyError, TypeError):
                        pass

        h_bbsc = _fix_qubits(h_bbsc, self._has_side_chain_second_bead)
        h_scbb = _fix_qubits(h_scbb, self._has_side_chain_second_bead)
        return h_bbsc, h_scbb

    def _create_h_scsc(self) -> Union[PauliSumOp, PauliOp]:
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        side chain (SC) beads. In the absence of side chains, this function
        returns a value of 0.

        Returns:
            Hamiltonian term consisting of side chain pairwise interactions
        """
        penalty_1 = self._penalty_parameters.penalty_1
        h_scsc = 0
        main_chain_len = len(self._peptide.get_main_chain)
        side_chain = self._peptide.get_side_chain_hot_vector()
        for i in range(1, main_chain_len - 3):
            for j in range(i + 5, main_chain_len + 1):
                if (j - i) % 2 == 0:
                    continue
                if side_chain[i - 1] == 0 or side_chain[j - 1] == 0:
                    continue
                h_scsc += self._contact_map.lower_side_upper_side[i][j] ^ (
                        self._distance_map._first_neighbor(
                            self._peptide, i, 1, j, 1, penalty_1, self._pair_energies
                        )
                        + self._distance_map._second_neighbor(
                    self._peptide, i, 1, j, 0, penalty_1, self._pair_energies
                )
                        + self._distance_map._second_neighbor(
                    self._peptide, i, 0, j, 1, penalty_1, self._pair_energies
                )
                )
        return _fix_qubits(h_scsc, self._has_side_chain_second_bead)

    def _create_h_short(self) -> Union[PauliSumOp, PauliOp]:
        """
        Creates Hamiltonian constituting interactions between beads that are no more than
        4 beads apart. If no side chains are present, this function returns 0.

        Returns:
            Contribution to energetic Hamiltonian for interactions between beads that are no more
            than 4 beads apart.
        """
        main_chain_len = len(self._peptide.get_main_chain)
        side_chain = self._peptide.get_side_chain_hot_vector()
        h_short = 0
        for i in range(1, main_chain_len - 2):
            # checks interactions between beads no more than 4 beads apart
            if side_chain[i - 1] == 1 and side_chain[i + 2] == 1:
                op1 = self._create_turn_operators(
                    self._peptide.get_main_chain[i + 1],
                    self._peptide.get_main_chain[i - 1].side_chain[0],
                )
                op2 = self._create_turn_operators(
                    self._peptide.get_main_chain[i - 1],
                    self._peptide.get_main_chain[i + 2].side_chain[0],
                )
                coeff = float(
                    self._pair_energies[i][1][i + 3][1]
                    + 0.1
                    * (self._pair_energies[i][1][i + 3][0] + self._pair_energies[i][0][i + 3][1])
                )
                composed = op1 @ op2
                h_short += (coeff * composed).reduce()
        h_short = _fix_qubits(h_short, self._has_side_chain_second_bead)

        return h_short
