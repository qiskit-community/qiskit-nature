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
from typing import Dict, List, Optional, Tuple

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError

from .initializer import Initializer


class MP2Initializer(Initializer):
    """
    An Initializer class for using the Moller-Plesset 2nd order (MP2) correction
    coefficients as an initial point for VQE when using UCC.

    An MP2Initializer object can be passed to UCC(SD) as an optional argument.
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
            reference_energy: The uncorrected Hartree-Fock energy.
            threshold: Computed coefficients and energy deltas will be set to
                       zero if their value is below this threshold.
        """
        super().__init__()

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
        """
        Returns:
            The number of molecular orbitals.
        """
        return self._num_orbitals

    @property
    def num_spin_orbitals(self) -> int:
        """
        Returns:
            The number of spin orbitals.
        """
        return self._num_orbitals * 2

    @property
    def energy_correction(self) -> float:
        """
        Returns:
            The MP2 delta energy correction for the molecule.
        """
        return self._energy_correction

    @property
    def energy_corrections(self) -> np.ndarray:
        """
        Returns:
            The MP2 delta energy corrections for each excitation.
        """
        return self._energy_corrections

    @property
    def absolute_energy(self) -> float:
        """
        Raises:
            QiskitNatureError: Raised if reference energy is not set.

        Returns:
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
        """
        Returns:
            The MP2 terms for the molecule.
        """
        return self._terms

    @property
    def coefficients(self) -> List[float]:
        """
        Returns:
            The MP2 coefficients for the molecule.
        """
        return self._coefficients

    @property
    def excitations(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """
        Returns:
            The excitations.
        """
        return [_string_to_tuple(key) for key in self._terms]

    def compute_corrections(
        self,
        excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the MP2 coefficient and energy corrections for each double excitation.

        Args:
            excitations : Sequence of excitations.

        Returns:
            Correction coefficients and energy corrections.
        """
        terms = {}
        for excitation in excitations:
            if len(excitation[0]) == 2:
                # MP2 needs double excitations.
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

    def _compute_correction(
        self, excitation: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> Tuple[float, float]:
        """Compute the MP2 coefficient and energy corrections given a double excitation.

        Each double excitation given by [i,a,j,b] has a coefficient computed using
            coeff = -(2 * Tiajb - Tibja)/(oe[b] + oe[a] - oe[i] - oe[j])
        where oe[] is the orbital energy.
        and an energy delta given by
            e_delta = coeff * Tiajb

        All the computations are done using the molecule orbitals but the indices used
        in the excitation information passed in and out are in the block spin orbital
        numbering as normally used by the nature module:
          - alpha runs from 0 to num_orbitals - 1
          - beta runs from num_orbitals to num_orbitals * 2 - 1

        Args:
            excitations: Sequence of excitations.

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


def _string_to_tuple(excitation_str: str) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Args:
        excitation_str: Excitations as a string.

    Returns:
        Excitations as a list of tuples.
    """
    return tuple(ast.literal_eval(excitation_str))
