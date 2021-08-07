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
"""Builds a distance map that stores distances between beads in a peptide."""
import collections
import logging
from typing import Dict, DefaultDict, Tuple, Union, List

from qiskit.opflow import OperatorBase, PauliSumOp, PauliOp

from ..peptide.beads.base_bead import BaseBead
from ..peptide.beads.main_bead import MainBead
from ..peptide.beads.side_bead import SideBead
from ..qubit_utils.qubit_fixing import _fix_qubits
from ..peptide.peptide import Peptide

logger = logging.getLogger(__name__)


class DistanceMapBuilder:
    """
    Distance Map Builder.
    """

    def __init__(self):
        self._distance_map_axes = self._init_dicts()
        self.num_distances = 0

    def _create_distance_qubits(
        self,
        peptide: Peptide,
    ) -> Tuple[DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]], int]:
        """
        Creates total distances between all bead pairs by summing the
        distances over all turns with axes, a = 0,1,2,3.

        Args:
            peptide: A Peptide object that includes all information about a protein.

        Returns:
            Tuple of beads-indexed dictionary that stores distances between beads of a peptide as
            qubit operators and the number of distances calculated.
        """
        self._calc_distances_main_chain(peptide)
        self._add_distances_side_chain(peptide)
        main_chain_len = len(peptide.get_main_chain)

        distance_map: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]] = collections.defaultdict(
            dict
        )

        for lower_bead_ind in range(1, main_chain_len):  # upper_bead_ind>lower_bead_ind
            for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
                lower_main_bead, lower_side_bead = self._get_main_and_side_beads(
                    lower_bead_ind, peptide
                )
                upper_main_bead, upper_side_bead = self._get_main_and_side_beads(
                    upper_bead_ind, peptide
                )

                distance_map[lower_main_bead][upper_main_bead] = self._calc_distance(
                    lower_main_bead,
                    upper_main_bead,
                )
                if distance_map[lower_main_bead][upper_main_bead] != 0:
                    self.num_distances += 1

                distance_map[lower_side_bead][upper_main_bead] = self._calc_distance(
                    lower_side_bead,
                    upper_main_bead,
                )
                if distance_map[lower_side_bead][upper_main_bead] != 0:
                    self.num_distances += 1

                distance_map[lower_main_bead][upper_side_bead] = self._calc_distance(
                    lower_main_bead,
                    upper_side_bead,
                )
                if distance_map[lower_main_bead][upper_side_bead] != 0:
                    self.num_distances += 1

                distance_map[lower_side_bead][upper_side_bead] = self._calc_distance(
                    lower_side_bead,
                    upper_side_bead,
                )
                if distance_map[lower_side_bead][upper_side_bead] != 0:
                    self.num_distances += 1

        logger.info(self.num_distances, " distances created")
        return distance_map, self.num_distances

    def _calc_distance(
        self,
        lower_bead: BaseBead,
        upper_bead: BaseBead,
    ) -> Union[PauliSumOp, PauliOp]:
        distance = 0
        for dist_map_ax in self._distance_map_axes:
            distance += dist_map_ax[lower_bead][upper_bead] ** 2
        return _fix_qubits(distance)

    def _calc_distances_main_chain(self, peptide: Peptide) -> None:
        # pylint:disable=anomalous-backslash-in-string
        """
        Calculates distance between beads based on the number of turns in
        the main chain. Note, here we consider distances between beads
        not on side chains. For a particular axis, a, we calculate the
        distance between lower_bead_ind and upper_bead_ind bead pairs,
        distance_map_axis_a :math:`= \sum_k (-1)^k*indica(k)` where :math:`k` iterates from
        lower_bead_ind to upper_bead_ind - 1.

        Args:
            peptide: A Peptide object that includes all information about a protein.
        """
        main_chain_len = len(peptide.get_main_chain)

        for lower_bead_ind in range(1, main_chain_len):
            for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
                lower_main_bead = peptide.get_main_chain[lower_bead_ind - 1]
                upper_main_bead = peptide.get_main_chain[upper_bead_ind - 1]

                for k in range(lower_bead_ind, upper_bead_ind):
                    indic_funs = peptide.get_main_chain[k - 1].indicator_functions
                    for dist_map_ax, indic_fun_x in zip(self._distance_map_axes, indic_funs):
                        dist_map_ax[lower_main_bead][upper_main_bead] += (-1) ** k * indic_fun_x

                for dist_map_ax in self._distance_map_axes:
                    dist_map_ax[lower_main_bead][upper_main_bead] = _fix_qubits(
                        dist_map_ax[lower_main_bead][upper_main_bead]
                    )

    @staticmethod
    def _init_dicts():
        return [
            collections.defaultdict(lambda: collections.defaultdict(int)),
            collections.defaultdict(lambda: collections.defaultdict(int)),
            collections.defaultdict(lambda: collections.defaultdict(int)),
            collections.defaultdict(lambda: collections.defaultdict(int)),
        ]

    def _add_distances_side_chain(self, peptide: Peptide) -> None:
        """
        Calculates distances between beads located on side chains and adds the contribution to the
        distance calculated between beads (lower_bead_ind and upper_bead_ind) on the main chain. In
        the absence of side chains, this function returns a value of 0.

        Args:
            peptide: A Peptide object that includes all information about a protein.
        """
        main_chain_len = len(peptide.get_main_chain)
        side_chain = peptide.get_side_chain_hot_vector()
        for lower_bead_ind in range(1, main_chain_len):  # upper_bead_ind>lower_bead_ind
            for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
                lower_main_bead, lower_side_bead = self._get_main_and_side_beads(
                    lower_bead_ind, peptide
                )
                upper_main_bead, upper_side_bead = self._get_main_and_side_beads(
                    upper_bead_ind, peptide
                )

                upper_indic_funs = self._get_indicator_funs(peptide, side_chain, upper_bead_ind)
                lower_indic_funs = self._get_indicator_funs(peptide, side_chain, lower_bead_ind)

                self._calc_dists_main_side_all_axes(
                    peptide,
                    lower_bead_ind,
                    lower_main_bead,
                    upper_bead_ind,
                    upper_side_bead,
                    upper_indic_funs,
                )

                self._calc_dists_side_main_all_axes(
                    peptide,
                    lower_bead_ind,
                    lower_side_bead,
                    upper_bead_ind,
                    upper_main_bead,
                    lower_indic_funs,
                )

                self._calc_dists_side_side_all_axes(
                    peptide,
                    lower_bead_ind,
                    lower_side_bead,
                    lower_indic_funs,
                    upper_bead_ind,
                    upper_side_bead,
                    upper_indic_funs,
                )

    @staticmethod
    def _get_main_and_side_beads(bead_ind: int, peptide: Peptide) -> Tuple[MainBead, SideBead]:
        main_bead = peptide.get_main_chain[bead_ind - 1]
        if main_bead.side_chain:
            side_bead = main_bead.side_chain[0]
        else:
            side_bead = None
        return main_bead, side_bead

    @staticmethod
    def _get_indicator_funs(
        peptide: Peptide, side_chain: List[bool], bead_ind: int
    ) -> Union[
        Tuple[None, None, None, None], Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase]
    ]:
        if side_chain[bead_ind - 1]:
            indic_0, indic_1, indic_2, indic_3 = (
                peptide.get_main_chain[bead_ind - 1].side_chain[0].indicator_functions
            )
        else:
            indic_0, indic_1, indic_2, indic_3 = None, None, None, None
        return indic_0, indic_1, indic_2, indic_3

    def _calc_dists_side_side_all_axes(
        self,
        peptide: Peptide,
        lower_bead_ind: int,
        lower_side_bead: BaseBead,
        lower_indic_funs: Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase],
        upper_bead_ind: int,
        upper_side_bead: BaseBead,
        upper_indic_funs: Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase],
    ) -> None:
        for dist_map_ax, lower_indic_fun_x, upper_indic_fun_x in zip(
            self._distance_map_axes, lower_indic_funs, upper_indic_funs
        ):

            dist_map_ax[lower_side_bead][upper_side_bead] = self._calc_distance_term(
                peptide,
                dist_map_ax,
                lower_bead_ind,
                lower_indic_fun_x,
                upper_bead_ind,
                upper_indic_fun_x,
            )

    def _calc_dists_side_main_all_axes(
        self,
        peptide: Peptide,
        lower_bead_ind: int,
        lower_side_bead: BaseBead,
        upper_bead_ind: int,
        upper_main_bead: BaseBead,
        indic_funs: Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase],
    ) -> None:
        for dist_map_ax, indic_fun_x in zip(self._distance_map_axes, indic_funs):
            dist_map_ax[lower_side_bead][upper_main_bead] = self._calc_distance_term(
                peptide, dist_map_ax, lower_bead_ind, indic_fun_x, upper_bead_ind, None
            )

    def _calc_dists_main_side_all_axes(
        self,
        peptide: Peptide,
        lower_bead_ind: int,
        lower_bead: BaseBead,
        upper_bead_ind: int,
        upper_bead: BaseBead,
        indic_funs: Tuple[OperatorBase, OperatorBase, OperatorBase, OperatorBase],
    ) -> None:
        for dist_map_ax, indic_fun_x in zip(self._distance_map_axes, indic_funs):

            dist_map_ax[lower_bead][upper_bead] = self._calc_distance_term(
                peptide, dist_map_ax, lower_bead_ind, None, upper_bead_ind, indic_fun_x
            )

    def _calc_distance_term(
        self,
        peptide: Peptide,
        distance_map_axis_x: Dict[BaseBead, OperatorBase],
        lower_bead_ind: int,
        lower_indic_fun: OperatorBase,
        upper_bead_ind: int,
        upper_indic_fun: OperatorBase,
    ) -> Union[PauliSumOp, PauliOp]:
        lower_main_bead = peptide.get_main_chain[lower_bead_ind - 1]
        upper_main_bead = peptide.get_main_chain[upper_bead_ind - 1]
        result = distance_map_axis_x[lower_main_bead][upper_main_bead]
        if lower_indic_fun is not None:
            result -= (-1) ** lower_bead_ind * lower_indic_fun
        if upper_indic_fun is not None:
            result += (-1) ** upper_bead_ind * upper_indic_fun

        return _fix_qubits(result)
