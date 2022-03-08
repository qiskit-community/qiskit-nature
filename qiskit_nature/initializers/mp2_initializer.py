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

""" MP2Initializer class """

import ast
import numpy as np
from typing import Dict, List, Optional, Tuple

from qiskit_nature.exceptions import QiskitNatureError

from .initializer import Initializer
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis


class MP2Initializer(Initializer):
    """
    An Initializer class for using the Moller-Plesset 2nd order (MP2) correction
    coefficients as an initial point for finding the Minimum Eigensolver.

    | Each double excitation given by [i,a,j,b] has a coefficient computed using
    |     coeff = -(2 * Tiajb - Tibja)/(oe[b] + oe[a] - oe[i] - oe[j])
    | where oe[] is the orbital energy

    | and an energy delta given by
    |     e_delta = coeff * Tiajb

    All the computations are done using the molecule orbitals but the indexes used
    in the excitation information passed in and out are in the block spin orbital
    numbering as normally used by the nature module.
    """

    def __init__(
        self,
        num_spin_orbitals: int,
        orbital_energies: np.ndarray,
        integral_matrix: np.ndarray,
        reference_energy: Optional[float] = None,
        threshold: float = 1e-12,
    ):
        """
        Args:
            num_spin_orbitals: Number of spin orbitals.
            orbital energies: Electric orbital energies.
            integral_matrix: Electronic double excitation integral matrix.
            electronic_energy: ElectronicEnergy to extract the electronic integral matrix,
                               orbital energies and reference energy.
            threshold: Computed coefficients and energy deltas will be set to
                       zero if their value is below this threshold.
        """
        # Since spins are the same drop to MO indexing
        self._num_orbitals = num_spin_orbitals // 2
        self._integral_matrix = integral_matrix
        self._orbital_energies = orbital_energies
        self._reference_energy = reference_energy
        self._threshold = threshold

        # Computed with specific excitation list.
        self._terms = None
        self._coefficients = None
        self._energy_correction = None
        self._energy_corrections = None

    @property
    def num_orbitals(self) -> int:
        """Returns:
        The number of molecular orbitals.
        """
        return self._num_orbitals

    @property
    def num_spin_orbitals(self) -> int:
        """Returns:
        The number of spin orbitals.
        """
        return self._num_orbitals * 2

    @property
    def energy_correction(self) -> float:
        """Returns:
        The MP2 delta energy correction for the molecule.
        """
        return self._energy_correction

    @property
    def energy_corrections(self) -> np.ndarray:
        """Returns:
        The MP2 delta energy corrections for each excitation.
        """
        return self._energy_corrections

    @property
    def absolute_energy(self) -> float:
        """Returns:
        The absolute MP2 energy for the molecule.
        """
        if self._reference_energy is None:
            raise QiskitNatureError("Reference energy not set.")
        return self._reference_energy + self._energy_correction

    @property
    def reference_energy(self) -> float:
        """Returns:
            The reference Hartree Fock energy for the molecule.
        """
        return self._reference_energy

    @reference_energy.setter
    def qubit_converter(self, energy: float) -> None:
        """Sets the reference Hartree Fock energy for the molecule.
        Args:
            energy: Reference energy value.
        """
        self._reference_energy = energy

    @property
    def terms(self) -> Dict[str, Tuple[float, float]]:
        """Returns:
        The MP2 terms for the molecule.
        """
        return self._terms

    @property
    def coefficients(self) -> List[float]:
        """Returns:
        "The MP2 coefficients for the molecule.
        """
        return self._coefficients

    @property
    def excitations(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Returns:
        The excitations.
        """
        return [_string_to_tuple(key) for key in self._terms.keys()]

    def compute_corrections(
        self,
        excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the MP2 coefficient and energy corrections for each double excitation.

        Excitations is a list of:
        [(initial_orbital_1, initial_orbital_2) (final_orbital_1, final_orbital_2), ...]

        Spin orbital indexing is in block spin format:
          - alpha runs from 0 to num_orbitals - 1
          - beta runs from num_orbitals to num_orbitals * 2 - 1

        Args:
            excitations : Sequence of excitations.

        Returns:
            Correction coefficients and energy corrections.
        """
        terms = {}
        for excitation in excitations:
            if len(excitation[0]) == 2:
                coeff, e_delta = self._compute_correction(excitation)
            else:
                coeff, e_delta = 0, 0

            terms[str(excitation)] = (coeff, e_delta)

        coeffs = np.asarray([value[0] for value in terms.values()])
        e_deltas = np.asarray([value[1] for value in terms.values()])

        self._terms = terms
        self._coefficients = coeffs
        self._energy_corrections = e_deltas
        self._energy_correction = sum(e_deltas)
        return coeffs, e_deltas

    def _compute_correction(self, excitation) -> Tuple[float, float]:
        """Compute the MP2 coefficient and energy corrections given a double excitation.

        Excitations is a list of:
        [(initial_orbital_1, initial_orbital_2) (final_orbital_1, final_orbital_2), ...]

        Spin orbital indexing is in block spin format:
          - alpha runs from 0 to num_orbitals - 1
          - beta runs from num_orbitals to num_orbitals * 2 - 1

        Args:
            excitations : Sequence of excitations.

        Returns:
            Correction coefficients and energy corrections.
        """
        i = excitation[0][0] % self._num_orbitals
        j = excitation[0][1] % self._num_orbitals
        a = excitation[1][0] % self._num_orbitals
        b = excitation[1][1] % self._num_orbitals

        tiajb = self._integral_matrix[i, a, j, b]
        tibja = self._integral_matrix[i, b, j, a]

        num = 2 * tiajb - tibja
        denom = (
            self._orbital_energies[b]
            + self._orbital_energies[a]
            - self._orbital_energies[i]
            - self._orbital_energies[j]
        )
        coeff = -num / denom
        coeff = coeff if abs(coeff) > self._threshold else 0
        e_delta = coeff * tiajb
        e_delta = e_delta if abs(e_delta) > self._threshold else 0

        return (coeff, e_delta)

    # def mp2_get_term_info(
    #     self, freeze_core: bool = False, orbital_reduction: list = None
    # ):
    #     """
    #     With a reduced active space the set of used excitations can be less than allowing
    #     all available excitations. Given a (sub)set of excitations in the space this will update
    #     the list of correlation coefficients and a list of correlation energies ordered as per
    #     the excitation list provided.

    #     Args:
    #         excitation: A list of excitations for which to get the coeff and e_delta
    #         freeze_core: Whether core orbitals are frozen or not
    #         orbital_reduction: An optional list of ints indicating removed orbitals

    #     Raises:
    #         ValueError: Excitation not present in mp2 terms
    #     """
    #     # TODO bypass this for now!
    #     # terms = self.mp2_terms(freeze_core, orbital_reduction)

    #     coeffs = []
    #     e_deltas = []
    #     for excitation in self._doubles:
    #         # if len(excitation) != 4:
    #         #     raise ValueError("Excitation entry must be of length 4")
    #         key = str(excitation)
    #         if key in terms:
    #             coeff, e_delta = terms[key]
    #             coeffs.append(coeff)
    #             e_deltas.append(e_delta)
    #         else:
    #             raise ValueError(f"Excitation {excitation} not present in mp2 terms")
    #     return coeffs, e_deltas

    # def compute_mp2(
    #     self,
    #     num_spin_orbitals: int,
    #     excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    # ) -> None:

    #     # doubles is a list of:
    #     # [(initial_orbital_1, initial_orbital_2) (final_orbital1, final_orbital2)]
    #     # Spin orbital indexing is in block spin format:
    #      #   alpha runs from 0 to num_orbitals-1
    #     #   beta runs from num_orbitals to num_orbitals*2-1

    #     doubles = _get_double_excitations(excitations)

    #     electronic_integral = self._electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
    #     ints = electronic_integral.get_matrix()

    #     o_e = self._electronic_energy.orbital_energies

    #     # Since spins are same drop to MO indexing
    #     self._num_orbitals = num_spin_orbitals // 2

    #     terms = {}
    #     mp2_delta = 0
    #     for double in doubles:
    #         i = double[0][0] % self._num_orbitals
    #         j = double[0][1] % self._num_orbitals
    #         a = double[1][0] % self._num_orbitals
    #         b = double[1][1] % self._num_orbitals

    #         tiajb = ints[i, a, j, b]
    #         tibja = ints[i, b, j, a]

    #         num = 2 * tiajb - tibja
    #         denom = o_e[b] + o_e[a] - o_e[i] - o_e[j]
    #         coeff = -num / denom
    #         coeff = coeff if abs(coeff) > self._threshold else 0
    #         e_delta = coeff * tiajb
    #         e_delta = e_delta if abs(e_delta) > self._threshold else 0

    #         terms[str(double)] = (coeff, e_delta)
    #         mp2_delta += e_delta

    #     self._doubles = doubles
    #     self._terms = terms
    #     self._mp2_delta = mp2_delta
    #     self._mp2_energy = self._electronic_energy.reference_energy + self._mp2_delta

    #     return

    # def mp2_terms(self, freeze_core: bool = False, orbital_reduction: list = None) -> Dict:
    #     """
    #     Gets the set of MP2 terms for the molecule taking into account index adjustments
    #     due to frozen core and/or other orbital reduction

    #     Args:
    #         freeze_core: Whether core orbitals are frozen or not
    #         orbital_reduction: An optional list of ints indicating removed orbitals

    #     Returns:
    #         A dictionary of excitations where the key is a string in the form
    #         from_to_from_to e.g. 0_4_6_10 and the value is a tuple of
    #         (coeff, e_delta)
    #     """
    #     if orbital_reduction is not None:
    #         raise NotImplementedError("Orbital reduction for mp2_terms is not implemented yet.")

    #     if freeze_core:
    #         raise NotImplementedError("Core freezing for mp2_terms is not implemented yet.")

    #     # orbital_reduction = orbital_reduction if orbital_reduction is not None else []

    #     # Compute the list of orbitals that will be removed. Here we do not care whether
    #     # it is occupied or not since the goal will be to subset the full set of excitation
    #     # terms, we originally computed, down to the set that exist within the remaining
    #     # orbitals.
    #     # core_list = self._core_orbitals if freeze_core else []
    #     # reduce_list = orbital_reduction
    #     # reduce_list = [x + self._num_orbitals if x < 0 else x for x in reduce_list]
    #     # remove_orbitals = sorted(set(core_list).union(set(reduce_list)))
    #     # remove_spin_orbitals = remove_orbitals + [x + self._num_orbitals for x in remove_orbitals]

    #     # An array of original indexes of the full set of spin orbitals. Plus an
    #     # array which will end up having the new indexes at the corresponding positions
    #     # of the original orbital after the removal has taken place. The original full
    #     # set will correspondingly have -1 values entered where orbitals have been removed
    #     full_spin_orbs = [*range(0, 2 * self._num_orbitals)]
    #     remain_spin_orbs = [-1] * len(full_spin_orbs)

    #     new_idx = 0
    #     for i, _ in enumerate(full_spin_orbs):
    #         # if full_spin_orbs[i] in remove_spin_orbitals:
    #         #     full_spin_orbs[i] = -1
    #         #     continue
    #         remain_spin_orbs[i] = new_idx
    #         new_idx += 1

    #     # Now we look through all the original excitations and check if all the from and to
    #     # values in the set or orbitals exists (is a subset of) the remaining orbitals in the
    #     # full spin set (note this now has -1 as value in indexes for which the orbital was
    #     # removed. If its a subset we remap the orbitals to the values that correspond to the
    #     # remaining spin orbital indexes.
    #     ret_terms = {}
    #     for k, v in self._terms.items():
    #         ex = _string_to_tuple(k)
    #         # FIXME this is hideous
    #         orbs = (ex[0][0], ex[0][1], ex[1][0], ex[1][1])
    #         if set(orbs) <= set(full_spin_orbs):
    #             new_idxs = [remain_spin_orbs[elem] for elem in orbs]
    #             coeff, e_delta = v
    #             ret_terms[str(new_idxs)] = (coeff, e_delta)

    #     return ret_terms


def _string_to_tuple(excitation_str: str) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    return tuple(ast.literal_eval(excitation_str))
