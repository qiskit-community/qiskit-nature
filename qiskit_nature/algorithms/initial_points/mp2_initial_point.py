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

"""Second-order Møller-Plesset perturbation theory (MP2) computation."""

from __future__ import annotations

import numpy as np
from qiskit_nature.exceptions import QiskitNatureError

from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.circuit.library import UCC
from qiskit_nature.properties.second_quantization.electronic.integrals.electronic_integrals import (
    ElectronicIntegrals,
)
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)

from .initial_point import InitialPoint


class MP2InitialPoint(InitialPoint):
    """Second-order Møller-Plesset perturbation theory (MP2) computation.

    A class for computing the second-order Møller-Plesset perturbation theory (MP2) corrections.
    This is intended for use as an initial point for the VQE parameters in combination with a UCC
    ansatz.

    :class:`MP2InitialPoint` requires two-body molecular orbital electronic integrals and orbital
    energies from
    :class:`qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`,
    which should be passed in via the :attr:`grouped_property` attribute.

    :class:`MP2InitialPoint` also requires the :attr:`excitation_list` from the :attr:`ansatz`
    to ensure that the coefficients are mapped correctly in the initial point array. This can be
    overwritten by setting :attr:`excitation_list` directly.

    The coefficients and energy corrections are computed using :meth:`compute`, which requires
    :attr:`grouped_property` and :attr:`ansatz` to be set or passed as arguments to
    :meth:`compute`.

    Following computation, the initial point array can be extracted via :meth:`to_numpy_array`.
    Array elements with indices corresponding to double excitations in the :attr:`excitation_list`
    will use the computed MP2 coefficient, while those that correspond to single, triple, or
    higher excitations will be zero.

    The array of energy corrections for each excitation can be
    recovered using :meth:`get_energy_corrections`. The overall correction can be obtained via
    :meth:`get_energy_correction`.

    If the Hartree-Fock reference energy was found in
    :class:`qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`, it will
    be used to compute the absolute MP2 energy via :meth:`get_energy`.
    """

    def __init__(self, threshold: float = 1e-12) -> None:
        super().__init__()
        self.threshold: float = threshold

        self._grouped_property: GroupedSecondQuantizedProperty | None = None
        self._integral_matrix: np.ndarray | None = None
        self._orbital_energies: np.ndarray | None = None
        self._reference_energy: float = 0.0

        self._ansatz: UCC | None = None
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None

        self._corrections: dict[str, tuple[float, float]] | None = None
        self._missing_input_error_message: str = (
            "Not enough information has been provided to compute the MP2 corrections. "
            "Set `grouped_property` and `ansatz` or call `compute` with them as arguments."
        )

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty:
        """The grouped property."""
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        """The grouped property.

        The grouped property is required to contain the ElectronicEnergy, which must contain
        the two-body molecular orbital matrix and the orbital energies. Optionally, it will also
        use the Hartree-Fock reference energy to compute the absolute energy.

        Raises:
            QiskitNatureError: If
                :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`
                is missing or the two-body molecular orbital matrix or the orbital energies are not
                found.
        """

        electronic_energy: ElectronicEnergy | None = grouped_property.get_property(ElectronicEnergy)
        if not isinstance(electronic_energy, ElectronicEnergy):
            raise QiskitNatureError(f"ElectronicEnergy not in grouped property: {grouped_property}")

        two_body_mo_integral: ElectronicIntegrals | None = (
            electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
        )
        if not isinstance(two_body_mo_integral, ElectronicIntegrals):
            raise QiskitNatureError(
                f"Two body MO electronic integral not in grouped property: {grouped_property}"
            )

        orbital_energies: np.ndarray | None = electronic_energy.orbital_energies
        if not isinstance(orbital_energies, np.ndarray):
            raise QiskitNatureError(f"Orbital energies not in grouped property: {grouped_property}")

        # Invalidate any previous computation.
        self._corrections = None

        self._integral_matrix = two_body_mo_integral.get_matrix()
        self._orbital_energies = orbital_energies
        self._reference_energy = electronic_energy.reference_energy if not None else 0.0
        self._grouped_property = grouped_property

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        """The UCC ansatz.

        This is used to ensure that the :attr:`excitation_list` matches with the UCC ansatz that
        will be used with the VQE algorithm.
        """
        if not isinstance(ansatz, UCC):
            raise QiskitNatureError(
                f"MP2InitialPoint requires a UCC ansatz, but got type: {type(ansatz)}"
            )

        # Operators must be built early to compute excitation list.
        _ = ansatz.operators

        # Invalidate any previous computation.
        self._corrections = None

        self._excitation_list = ansatz.excitation_list
        self._ansatz = ansatz

    @property
    def threshold(self) -> float:
        """The energy threshold for MP2 computation.

        Computed initial point and energy corrections will be set to zero if their absolute value
        is below this threshold.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """The energy threshold for the MP2 computation.

        Computed coefficients and energy corrections will be set to zero if their absolute value is
        below this threshold.
        """
        try:
            threshold = abs(float(threshold))
        except TypeError:
            threshold = 0.0

        # Invalidate any previous computation.
        self._corrections = None

        self._threshold = float(threshold)

    @property
    def excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The list of excitations."""
        return self._excitation_list

    @excitation_list.setter
    def excitation_list(self, excitations: list[tuple[tuple[int, ...], tuple[int, ...]]]):
        """The list of excitations.

        Setting this will overwrite the excitation list from the ansatz.
        """
        # Invalidate any previous computation.
        self._corrections = None

        self._excitation_list = excitations

    def to_numpy_array(self) -> np.ndarray:
        """Convert the computed initial point to a numpy array.

        Raises:
            QiskitNatureError: If the initial point has not and cannot yet be computed.

        Returns:
            The initial point as a numpy array of MP2 correction coefficients.
        """
        if self._corrections is None:
            if self._grouped_property is None or self._ansatz is None:
                raise QiskitNatureError(self._missing_input_error_message)
            self.compute()
        return np.asarray([value[0] for value in self._corrections.values()])

    def compute(
        self,
        grouped_property: GroupedSecondQuantizedProperty | None = None,
        ansatz: UCC | None = None,
    ) -> None:
        """Compute the second-order Møller-Plesset perturbation theory (MP2) corrections.

        This is intended for use as an initial point for the VQE parameters in combination with a UCC
        ansatz.

        :meth:`~compute` requires two-body molecular orbital electronic integrals and orbital
        energies from
        :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`,
        which should be passed in via the :attr:`grouped_property` attribute.

        :meth:`compute` also requires the :attr:`excitation list` from the :attr:`~ansatz` to
        ensure that the coefficients are mapped correctly in the initial point array. This can be
        overwritten by setting :attr:`~excitation_list` directly.

        :attr:`grouped_property` and :attr:`~ansatz` can be set externally or passed as arguments
        to :meth:`~compute`.

        Args:
            grouped_property: A grouped second-quantized property that is required to contain the
                :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`.
                From this we require the two-body molecular orbital electronic integrals and orbital
                energies.
            ansatz: The UCC ansatz. Required to set the :attr:`excitation list` to ensure that the
            coefficients are mapped correctly in the initial point array.

        Raises:
            QiskitNatureError: If :attr:`grouped_property` and/or :attr`ansatz` are not set.
        """
        if isinstance(grouped_property, GroupedSecondQuantizedProperty):
            self.grouped_property = grouped_property
        else:
            if not isinstance(self._grouped_property, GroupedSecondQuantizedProperty):
                raise QiskitNatureError("Cannot compute. `grouped_property` has not been set.")

        if isinstance(ansatz, UCC):
            self.ansatz = ansatz
        else:
            if not isinstance(self._excitation_list, list):
                raise QiskitNatureError(
                    "Cannot compute. `excitation_list` has not been set directly or via `ansatz`."
                )

        self._corrections = self._compute_corrections()

    def _compute_corrections(
        self,
    ) -> dict[str, tuple[float, float]]:
        """Compute the MP2 coefficients and energy corrections.

        Tuples of the coefficients and energy corrections are stored in a dictionary with a string
        of the corresponding excitation. This dictionary isn't directly exposed to the user, but is
        retained for internal clarity and validation. Non-double excitations will have zero
        coefficient and energy correction.

        Returns:
            Dictionary with MP2 coefficients and energy_corrections for each excitation.
        """
        corrections = {}
        for excitation in self._excitation_list:
            if len(excitation[0]) == 2:
                # Compute MP2 corrections using double excitations.
                coefficient, energy_correction = self._compute_correction(excitation)
            else:
                # No corrections for single, triple, and higher excitations.
                coefficient, energy_correction = 0.0, 0.0

            corrections[str(excitation)] = (coefficient, energy_correction)

        return corrections

    def _compute_correction(
        self, excitation: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[float, float]:
        """Compute the MP2 coefficient and energy correction given a double excitation.

        Each double excitation indexed by ::math`i,a,j,b` has a correction coefficient,

        ..math::
            c_{i,a,j,b} = -\\frac{2 T_{i,a,j,b} - T_{i,b,j,a}}{E_b + E_a - E_i - E_j},

        where ::math::`E` is the orbital energy and ::math::`T` is the integral matrix.
        And an energy correction,

        ..math..
            correction = E_{i, a, j, b} = c_{i,a,j,b} T_{i,a,j,b}.

        These computations use molecular orbitals, but the indices used in the excitation
        information passed in and out use the block spin orbital numbering common to Qiskit Nature:
          - ::math::`\\alpha` runs from ``0`` to ``num_molecular_orbitals - 1``,
          - ::math::`'\\'beta` runs from ``num_molecular_orbitals`` to
            ``num_molecular_orbitals * 2 - 1``.

        Args:
            excitations: List of excitations.

        Returns:
            Coefficient and energy correction for a given double excitation.
        """
        integral_matrix = self._integral_matrix
        orbital_energies = self._orbital_energies
        threshold = self._threshold

        # Infer the number of molecular orbitals from the MO matrix.
        num_molecular_orbitals: int = integral_matrix.shape[0]

        i = excitation[0][0] % num_molecular_orbitals
        j = excitation[0][1] % num_molecular_orbitals
        a = excitation[1][0] % num_molecular_orbitals
        b = excitation[1][1] % num_molecular_orbitals

        expectation_value_iajb = integral_matrix[i, a, j, b]
        expectation_value_ibja = integral_matrix[i, b, j, a]

        expectation_value = 2 * expectation_value_iajb - expectation_value_ibja
        orbital_energy_correction = (
            orbital_energies[b] + orbital_energies[a] - orbital_energies[i] - orbital_energies[j]
        )
        correction_coeff = -expectation_value / orbital_energy_correction
        correction_coeff = correction_coeff if abs(correction_coeff) > threshold else 0.0

        energy_correction = correction_coeff * expectation_value_iajb
        energy_correction = energy_correction if abs(energy_correction) > threshold else 0.0

        return correction_coeff, energy_correction

    def get_energy_corrections(self) -> np.ndarray:
        """The individual MP2 energy corrections for each excitation.

        Raises:
            QiskitNatureError: If the initial point has not and cannot yet be computed.

        Returns:
            The individual MP2 energy corrections for each excitation.
        """
        if self._corrections is None:
            if self._grouped_property is None or self._ansatz is None:
                raise QiskitNatureError(self._missing_input_error_message)
            self.compute()
        return np.asarray([value[1] for value in self._corrections.values()])

    def get_energy_correction(self) -> float:
        """Returns the overall MP2 energy correction for the molecule."""
        return self.get_energy_corrections().sum()

    def get_energy(self) -> float:
        """Returns the absolute MP2 energy for the molecule.

        If the reference energy has not been recovered from
        :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`
        this will be equal to :meth:`get_energy_corrrection`.
        """
        return self._reference_energy + self.get_energy_correction()
