# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" MP2Info class """

from typing import Dict

from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis


class MP2Info:
    """
    A utility class for Moller-Plesset 2nd order (MP2) information

    | Each double excitation given by [i,a,j,b] has a coefficient computed using
    |     coeff = -(2 * Tiajb - Tibja)/(oe[b] + oe[a] - oe[i] - oe[j])
    | where oe[] is the orbital energy

    | and an energy delta given by
    |     e_delta = coeff * Tiajb

    All the computations are done using the molecule orbitals but the indexes used
    in the excitation information passed in and out are in the block spin orbital
    numbering as normally used by the nature module.
    """

    def __init__(self, electronic_energy, threshold: float = 1e-12):
        """
        A utility class for MP2 info

        Args:
            problem: ElectronicStructureProblem from chemistry driver
            threshold: Computed coefficients and energy deltas will be set to
                       zero if their value is below this threshold
        """
        self._electronic_energy = electronic_energy
        self._threshold = threshold
        # self._mp2_energy = problem.hf_energy + self._mp2_delta
        # self._core_orbitals = problem.core_orbitals

    @property
    def mp2_delta(self):
        """
        Get the MP2 delta energy correction for the molecule

        Returns:
             float: The MP2 delta energy
        """
        return self._mp2_delta

    #     @property
    #     def mp2_energy(self):
    #         """
    #         Get the MP2 energy for the molecule

    #         Returns:
    #             float: The MP2 energy
    #         """
    #         return self._mp2_energy

    #     def mp2_terms(self, freeze_core: bool = False, orbital_reduction: list = None) -> Dict:
    #         """
    #         Gets the set of MP2 terms for the molecule taking into account index adjustments
    #         due to frozen core and/or other orbital reduction

    #         Args:
    #             freeze_core: Whether core orbitals are frozen or not
    #             orbital_reduction: An optional list of ints indicating removed orbitals

    #         Returns:
    #             A dictionary of excitations where the key is a string in the form
    #             from_to_from_to e.g. 0_4_6_10 and the value is a tuple of
    #             (coeff, e_delta)
    #         """
    #         orbital_reduction = orbital_reduction if orbital_reduction is not None else []

    #         # Compute the list of orbitals that will be removed. Here we do not care whether
    #         # it is occupied or not since the goal will be to subset the full set of excitation
    #         # terms, we originally computed, down to the set that exist within the remaining
    #         # orbitals.
    #         core_list = self._core_orbitals if freeze_core else []
    #         reduce_list = orbital_reduction
    #         reduce_list = [x + self._num_orbitals if x < 0 else x for x in reduce_list]
    #         remove_orbitals = sorted(set(core_list).union(set(reduce_list)))
    #         remove_spin_orbitals = remove_orbitals + [x + self._num_orbitals for x in remove_orbitals]

    #         # An array of original indexes of the full set of spin orbitals. Plus an
    #         # array which will end up having the new indexes at the corresponding positions
    #         # of the original orbital after the removal has taken place. The original full
    #         # set will correspondingly have -1 values entered where orbitals have been removed
    #         full_spin_orbs = [*range(0, 2 * self._num_orbitals)]
    #         remain_spin_orbs = [-1] * len(full_spin_orbs)

    #         new_idx = 0
    #         for i, _ in enumerate(full_spin_orbs):
    #             if full_spin_orbs[i] in remove_spin_orbitals:
    #                 full_spin_orbs[i] = -1
    #                 continue
    #             remain_spin_orbs[i] = new_idx
    #             new_idx += 1

    #         # Now we look through all the original excitations and check if all the from and to
    #         # values in the set or orbitals exists (is a subset of) the remaining orbitals in the
    #         # full spin set (note this now has -1 as value in indexes for which the orbital was
    #         # removed. If its a subset we remap the orbitals to the values that correspond to the
    #         # remaining spin orbital indexes.
    #         ret_terms = {}
    #         for k, v in self._terms.items():
    #             orbs = _str_to_list(k)
    #             if set(orbs) <= set(full_spin_orbs):
    #                 new_idxs = [remain_spin_orbs[elem] for elem in orbs]
    #                 coeff, e_delta = v
    #                 ret_terms[_list_to_str(new_idxs)] = (coeff, e_delta)

    #         return ret_terms

    #     def mp2_get_term_info(
    #         self, excitation_list: list, freeze_core: bool = False, orbital_reduction: list = None
    #     ):
    #         """
    #         With a reduced active space the set of used excitations can be less than allowing
    #         all available excitations. Given a (sub)set of excitations in the space this will return
    #         a list of correlation coefficients and a list of correlation energies ordered as per
    #         the excitation list provided.

    #         Args:
    #             excitation_list: A list of excitations for which to get the coeff and e_delta
    #             freeze_core: Whether core orbitals are frozen or not
    #             orbital_reduction: An optional list of ints indicating removed orbitals

    #         Returns:
    #             Tuple(list, list): List of coefficients and list of energy deltas

    #         Raises:
    #             ValueError: Excitation not present in mp2 terms
    #         """
    #         terms = self.mp2_terms(freeze_core, orbital_reduction)
    #         coeffs = []
    #         e_deltas = []
    #         for excitation in excitation_list:
    #             if len(excitation) != 4:
    #                 raise ValueError("Excitation entry must be of length 4")
    #             key = _list_to_str(excitation)
    #             if key in terms:
    #                 coeff, e_delta = terms[key]
    #                 coeffs.append(coeff)
    #                 e_deltas.append(e_delta)
    #             else:
    #                 raise ValueError("Excitation {} not present in mp2 terms".format(excitation))
    #         return coeffs, e_deltas

    def compute_mp2(
        self,
        num_spin_orbitals,
        excitations,
    ) -> None:
        terms = {}
        mp2_delta = 0

        # Orbital indices given by this method are numbered according to the blocked spin ordering
        doubles = _get_double_excitations(excitations)

        electronic_integral = self._electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
        ints = electronic_integral.get_matrix()

        o_e = self._electronic_energy.orbital_energies

        # Since spins are same drop to MO indexing
        self._num_orbitals = num_spin_orbitals // 2

        # doubles is list of [from, to, from, to] in spin orbital indexing where alpha runs
        # from 0 to num_orbitals-1, and beta from num_orbitals to num_orbitals*2-1
        for double in doubles:
            i = double[0][0] % self._num_orbitals
            j = double[0][1] % self._num_orbitals
            a_i = double[1][0] % self._num_orbitals
            b = double[1][1] % self._num_orbitals

            tiajb = ints[i, a_i, j, b]
            tibja = ints[i, b, j, a_i]

            num = 2 * tiajb - tibja
            denom = o_e[b] + o_e[a_i] - o_e[i] - o_e[j]
            coeff = -num / denom
            coeff = coeff if abs(coeff) > self._threshold else 0
            e_delta = coeff * tiajb
            e_delta = e_delta if abs(e_delta) > self._threshold else 0

            terms[_list_to_str(double)] = (coeff, e_delta)
            mp2_delta += e_delta

        self._terms = terms
        self._mp2_delta = mp2_delta

        return


def _list_to_str(idxs):
    return "_".join([str(x) for x in idxs])


def _str_to_list(str_idxs):
    return [int(x) for x in str_idxs.split("_")]


def _get_double_excitations(excitations: list) -> list:
    return list(filter(lambda excitation: len(excitation[0]) == 2, excitations))
