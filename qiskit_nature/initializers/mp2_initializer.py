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
from typing import Dict, List, Tuple

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis


class MP2Initializer:
    """Moller-Plesset Initializer.

    An initial point generator class for using the Moller-Plesset 2nd order (MP2)
    corrections as an initial point for VQE when using UCC(SD).

    An MP2Initializer object can be passed to UCC(SD) as an optional argument.
    This ensures that the same excitations are used to compute the corrections.
    """

    def __init__(
        self,
        num_spin_orbitals: int,
        electronic_energy: ElectronicEnergy,
        excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
        threshold: float = 1e-12,
    ) -> None:
        """
        Args:
            num_spin_orbitals: Number of spin orbitals.
            electronic_energy: Electronic energy grouped property.
            excitation_list: Sequence of excitations.
            reference_energy: The uncorrected Hartree-Fock energy.
            threshold: Computed initial point and energy deltas will be set to
                       zero if their value is below this threshold.
        """
        super().__init__()

        # Since spins are the same drop to MO indexing
        self._num_orbitals = num_spin_orbitals // 2
        self._integral_matrix = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()
        self._orbital_energies = electronic_energy.orbital_energies
        self._reference_energy = electronic_energy.reference_energy
        self._excitations = excitations
        self._threshold = threshold

        # Computed terms with a specific excitation list.
        terms = self._compute_corrections()

        self._terms = terms
        self._excitations = terms.keys()
        self._initial_point = np.asarray([value[0] for value in terms.values()])
        self._e_deltas = np.asarray([value[1] for value in terms.values()])
        self._e_delta = sum(self._e_deltas)

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
    def energy_delta(self) -> float:
        """
        Returns:
            The MP2 delta energy correction for the molecule.
        """
        return self._energy_delta

    @property
    def threshold(self) -> float:
        """
        Returns:
            The energy threshold.
        """
        return self._threshold

    @property
    def energy_deltas(self) -> np.ndarray:
        """
        Returns:
            The MP2 energy correction deltas for each excitation.
        """
        return self._energy_deltas

    @property
    def absolute_energy(self) -> float:
        """
        Raises:
            QiskitNatureError: If reference energy is not set.

        Returns:
            The absolute MP2 energy for the molecule.
        """
        if self._reference_energy is None:
            raise QiskitNatureError("Reference energy not set.")
        return self._reference_energy + self._energy_delta

    @property
    def reference_energy(self) -> float:
        """
        Returns:
            The reference Hartree-Fock energy for the molecule.
        """
        return self._reference_energy

    @property
    def terms(self) -> Dict[str, Tuple[float, float]]:
        """
        Returns:
            The MP2 terms for the molecule.
        """
        return self._terms

    @property
    def excitations(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """
        Returns:
            Sequence of excitations.
        """
        return [_string_to_tuple(key) for key in self._terms]

    @property
    def initial_point(self) -> List[float]:
        """
        Returns:
            The MP2 coefficients as an initial_point.
        """
        return self._initial_point

    def _compute_corrections(
        self,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute the MP2 coefficients and energy corrections for each double excitation.

        Args:
            excitations: Sequence of excitations.

        Returns:
            Initial point with MP2 coefficients for doubles and zero otherwise.
        """
        terms = {}
        for excitation in self._excitations:
            if len(excitation[0]) == 2:
                # MP2 needs double excitations.
                coeff, e_delta = self._compute_correction(excitation)
            else:
                # Leave single excitations unchanged.
                coeff, e_delta = 0.0, 0.0

            terms[str(excitation)] = (coeff, e_delta)

        return terms

    def _compute_correction(
        self, excitation: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> Tuple[float, float]:
        """Compute the MP2 coefficient and energy corrections given a double excitation.

        Each double excitation given by [i,a,j,b] has a coefficient computed using
            coeff = -(2 * T[i,a,j,b] - T[i,b,j,a)/(E[b] + E[a] - E[i] - E[j])
        where E is the orbital energy.
        and an energy delta given by
            e_delta = coeff * T[i,a,j,b]

        All the computations are done using the molecule orbitals but the indices used
        in the excitation information passed in and out are in the block spin orbital
        numbering as normally used by the nature module:
          - alpha runs from 0 to num_orbitals - 1
          - beta runs from num_orbitals to num_orbitals * 2 - 1

        Args:
            excitations: Sequence of excitations.

        Returns:
            Correction coefficients and energy deltas.
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
