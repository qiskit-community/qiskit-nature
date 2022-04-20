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

"""The MP2InitialPoint class to compute an initial point for the VQE Ansatz parameters."""

from __future__ import annotations

import numpy as np
from qiskit_nature.exceptions import QiskitNatureError

from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.circuit.library import UCC
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)

from .initial_point import InitialPoint


class MP2InitialPoint(InitialPoint):
    """Moller-Plesset 2nd order initial point generator.

    A class for computing the Moller-Plesset second order (MP2) corrections to use as an initial
    point with a VQE algorithm alongside a UCC ansatz.

    The MP2 calculation requires the two-body molecular orbital matrix, the orbital energies,
    and the Hartree-Fock reference energy from the ``ElectronicEnergy`` in the grouped
    property.

    An MP2InitialPoint uses the same excitations as generated by the UCC ansatz to ensure that
    the coefficients are mapped correctly in the initial point. Only initial point values that
    correspond to double-excitations will be non-zero.

    The coefficients and energy deltas are computed using the ``get_initial_point`` method.
    Thereafter, they can be accessed via the ``initial_point`` and ``energy_deltas`` properties.
    """

    def __init__(self) -> None:
        super().__init__()
        self.threshold = 1e-12
        self._corrections: dict[str, tuple[float, float]] = None

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty:
        """The grouped property."""
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        """The grouped property.

        Requires the two-body molecular orbital matrix, the orbital energies,
        and the Hartree-Fock reference energy from ``ElectronicEnergy``.

        Raises:
            QiskitNatureError: If the ``ElectronicEnergy`` is missing or the two-body molecular
            orbital matrix, the orbital energies, and/or the Hartree-Fock reference energy are
            missing from ``ElectronicEnergy``.
        """
        error_message = (
            f"The grouped property is required to contain the ElectronicEnergy, which must contain "
            f"the two-body molecular orbital matrix, the orbital energies, and the Hartree-Fock "
            f"reference energy. Got grouped property: {grouped_property}"
        )
        try:
            # Try to get the ElectronicEnergy from the grouped property.
            electronic_energy: ElectronicEnergy = grouped_property.get_property(ElectronicEnergy)

            # Get the two-body molecular orbital (MO) matrix.
            self._integral_matrix: np.ndarray = electronic_energy.get_electronic_integral(
                ElectronicBasis.MO, 2
            ).get_matrix()

            # Infer the number of molecular orbitals from the MO matrix.
            self._num_molecular_orbitals: int = self._integral_matrix.shape[0]

            # Get the orbital energies and Hartree-Fock reference energy.
            self._orbital_energies: np.ndarray = electronic_energy.orbital_energies
            self._reference_energy: float = electronic_energy.reference_energy
        except (TypeError):
            raise QiskitNatureError(error_message)

        if self._orbital_energies is None or self._reference_energy is None:
            raise QiskitNatureError(error_message)

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        """The UCC ansatz."""
        # Operators must be built early to compute excitation list.
        _ = ansatz.operators
        self.excitations = ansatz.excitation_list
        self._ansatz = ansatz

    @property
    def threshold(self) -> float:
        """The energy threshold for MP2 computation.

        Computed initial point and energy deltas will be set to zero if their absolute value is
        below this threshold.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """The energy threshold for MP2 computation.

        Computed initial point and energy deltas will be set to zero if their absolute value is
        below this threshold.
        """
        if threshold < 0.0:
            raise ValueError("The energy threshold cannot be negative.")
        self._threshold = threshold

    @property
    def excitations(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The list of excitations."""
        return self._excitations

    @excitations.setter
    def excitations(self, excitations: list[tuple[tuple[int, ...], tuple[int, ...]]]):
        """The list of excitations.

        If set externally, this will overwrite the excitation list from the ansatz."""
        self._excitations = excitations

    @property
    def num_molecular_orbitals(self) -> int:
        """The number of molecular orbitals.

        This is inferred from the shape from the molecular orbital matrix extracted from the
        ElectronicEnergy.
        """
        # Read-only access is provided for verification by the user.
        return self._num_molecular_orbitals

    @property
    def initial_point(self) -> np.ndarray:
        """The MP2 coefficients as an initial_point."""
        return np.asarray([val[0] for val in self._corrections.values()])

    @property
    def energy_deltas(self) -> np.ndarray:
        """The MP2 energy correction deltas for each excitation."""
        return np.asarray([val[1] for val in self._corrections.values()])

    @property
    def energy_delta(self) -> float:
        """The MP2 delta energy correction for the molecule."""
        return sum(self.energy_deltas)

    @property
    def energy(self) -> float:
        """The absolute MP2 energy for the molecule."""
        return self._reference_energy + self.energy_delta

    def get_initial_point(
        self,
        grouped_property: GroupedSecondQuantizedProperty | None = None,
        ansatz: UCC | None = None,
    ) -> np.ndarray:
        """Computes an MP2-informed initial point for the VQE algorithm.

        Computes the Moller-Plesset second order (MP2) corrections to use as an initial point with
        a VQE algorithm alongside a UCC ansatz.

        The MP2 calculation requires the two-body molecular orbital matrix, the orbital energies,
        and the Hartree-Fock reference energy from the ``ElectronicEnergy`` in the grouped
        property.

        An MP2InitialPoint uses the same excitations as generated by the UCC ansatz to ensure that
        the coefficients are mapped correctly in the initial point. Only initial point values that
        correspond to double-excitations will be non-zero.

        Args:
            grouped_property: A grouped second-quantized property that is required to contain the
                ``ElectronicEnergy``. Additionally, ``ElectronicEnergy`` is required to contain the
                two-body molecular orbital matrix, the orbital energies, and the Hartree-Fock
                reference energy. If this has already been set, it doesn't need to be passed again
                as an argument. If it is passed, it will overwrite the previous grouped property.
                If it is not passed and has not been set an error will be raised.
            ansatz: The UCC ansatz. If this has already been set, it doesn't need to be passed as
                an argument. If it is passed, it will overwrite the previous grouped property. If
                it is not passed and has not been set an error will be raised.

        Raises:
            QiskitNatureError: If ``grouped_property`` and/or ``ansatz`` are not set.

        Return:
            The computed initial point.
        """
        error_message = (
            "Set this property of MP2InitialPoint or pass as an argument to get_initial_point."
        )
        if grouped_property is None:
            if self._grouped_property is None:
                raise QiskitNatureError(f"The grouped property cannot be None. {error_message}")
        else:
            self.grouped_property = grouped_property

        if ansatz is None:
            if self._ansatz is None:
                raise QiskitNatureError(f"The ansatz cannot be None. {error_message}")
        else:
            self.ansatz = ansatz

        self._corrections = self._compute_corrections()
        return self.initial_point

    def _compute_corrections(
        self,
    ) -> dict[str, tuple[float, float]]:
        """Compute the MP2 coefficients and energy corrections for each double excitation.

        Tuples of the coefficients and energy deltas are stored in a dictionary with a string of
        the corresponding excitation. This dictionary isn't directly exposed to the user, but is
        retained for internal clarity and validation. Non-double excitations will have zero
        coefficient and energy delta.

        Returns:
            Dictionary with MP2 coefficients and energy_deltas for each excitation.
        """
        corrections = {}
        for excitation in self._excitations:
            if len(excitation[0]) == 2:
                # Compute MP2 corrections using double excitations.
                coeff, e_delta = self._compute_correction(excitation)
            else:
                # No corrections for single, triple, and higher excitations.
                coeff, e_delta = 0.0, 0.0

            corrections[str(excitation)] = (coeff, e_delta)

        return corrections

    def _compute_correction(
        self, excitation: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[float, float]:
        """Compute the MP2 coefficient and energy corrections given a double excitation.

        Each double excitation indexed by ::math`i,a,j,b` has a correction coefficient,

        ..math::
            c_{i,a,j,b} = -\\frac{2 T_{i,a,j,b} - T_{i,b,j,a}}{E_b + E_a - E_i - E_j},

        where ::math::`E` is the orbital energy and ::math::`T` is the integral matrix.
        And an energy delta,

        ..math..
            \\Delta E_{i, a, j, b} = c_{i,a,j,b} T_{i,a,j,b}.

        These computations use molecular orbitals, but the indices used in the excitation
        information passed in and out use the block spin orbital numbering common to Qiskit Nature:
          - ::math::`\\alpha` runs from ``0`` to ``num_molecular_orbitals - 1``,
          - ::math::`\\beta` runs from ``num_molecular_orbitals`` to
            ``num_molecular_orbitals * 2 - 1``.

        Args:
            excitations: List of excitations.

        Returns:
            Correction coefficients and energy deltas.
        """
        num_molecular_orbitals = self._num_molecular_orbitals
        integral_matrix = self._integral_matrix
        orbital_energies = self._orbital_energies
        threshold = self._threshold

        i = excitation[0][0] % num_molecular_orbitals
        j = excitation[0][1] % num_molecular_orbitals
        a = excitation[1][0] % num_molecular_orbitals
        b = excitation[1][1] % num_molecular_orbitals

        expectation_value_iajb = integral_matrix[i, a, j, b]
        expectation_value_ibja = integral_matrix[i, b, j, a]

        expectation_value = 2 * expectation_value_iajb - expectation_value_ibja
        orbital_energy_delta = (
            orbital_energies[b] + orbital_energies[a] - orbital_energies[i] - orbital_energies[j]
        )
        correction_coeff = -expectation_value / orbital_energy_delta
        correction_coeff = correction_coeff if abs(correction_coeff) > threshold else 0.0

        energy_delta = correction_coeff * expectation_value_iajb
        energy_delta = energy_delta if abs(energy_delta) > threshold else 0.0

        return correction_coeff, energy_delta
