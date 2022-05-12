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

from dataclasses import dataclass

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

from .hf_initial_point import HFInitialPoint


@dataclass(frozen=True)
class _Correction:
    """Data class for storing corrections."""

    excitation: tuple[tuple[int, ...], tuple[int, ...]]
    coefficient: float = 0.0
    energy: float = 0.0


class MP2InitialPoint(HFInitialPoint):
    """Compute the second-order Møller-Plesset perturbation theory (MP2) initial point.

    The computed MP2 correction coefficients are intended for use as an initial point for the VQE
    parameters in combination with a :class:`~qiskit_nature.circuit.library.ansatzes.ucc.UCC`
    ansatz.

    The coefficients and energy corrections are computed using the :meth:`compute` method, which
    requires the :attr:`grouped_property` and :attr:`ansatz` to be passed as arguments or the
    :attr:`grouped_property` and :attr:`excitation_list` attributes to be set already.

    ``MP2InitialPoint`` requires the
    :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`, which should
    be passed in via the :attr:`grouped_property` attribute. From this it must obtain the two-body
    molecular orbital electronic integrals and orbital energies. If the Hartree-Fock reference
    energy is also obtained, it will be used to compute the absolute MP2 energy using the
    :meth:`get_energy` method.

    ``MP2InitialPoint`` also requires the :attr:`excitation_list` from the :attr:`ansatz` to ensure
    that the coefficients map correctly to the initial point array. However, this can be substituted
    by setting the :attr:`excitation_list` attribute directly.

    Following computation, the initial point array can be extracted via the :meth:`to_numpy_array`
    method. The array of energy corrections indexed by excitation can be recovered using the
    :meth:`get_energy_corrections` method. The overall energy correction can be obtained via the
    :meth:`get_energy_correction` method.

    Coefficient and energy correction array elements with indices corresponding to double
    excitations in the :attr:`excitation_list` will have a value corresponding to the appropriate
    MP2 energy correction while those that correspond to single, triple, or higher excitations will
    have zero value.
    """

    def __init__(self, threshold: float = 1e-12) -> None:
        super().__init__()
        self.threshold: float = threshold
        self._integral_matrix: np.ndarray | None = None
        self._orbital_energies: np.ndarray | None = None
        self._corrections: list[_Correction] | None = None

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty | None:
        """The grouped property.

        The grouped property is required to contain the
        :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`, which
        must contain the two-body molecular orbitals matrix and the orbital energies. Optionally,
        it will also use the Hartree-Fock reference energy to compute the absolute energy.

        Raises:
            QiskitNatureError: If
                :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`
                is missing or the two-body molecular orbitals matrix or the orbital energies are not
                found.
            NotImplementedError: If alpha and beta spin molecular orbitals are not identical.
        """
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
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
        if not np.allclose(self._integral_matrix, two_body_mo_integral.get_matrix(2)):
            raise NotImplementedError(
                "MP2InitialPoint only supports restricted-spin setups. "
                "Alpha and beta spin orbitals must be identical. "
                "See https://github.com/Qiskit/qiskit-nature/issues/645."
            )

        self._orbital_energies = orbital_energies
        self._reference_energy = electronic_energy.reference_energy if not None else 0.0
        self._grouped_property = grouped_property

    @property
    def threshold(self) -> float:
        """The energy threshold for MP2 corrections.

        Computed coefficients and energy corrections will be set to zero if their absolute value is
        below this threshold's absolute value.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        try:
            threshold = abs(float(threshold))
        except TypeError:
            threshold = 0.0

        # Invalidate any previous computation.
        self._corrections = None

        self._threshold = threshold

    def compute(
        self,
        ansatz: UCC | None = None,
        grouped_property: GroupedSecondQuantizedProperty | None = None,
    ) -> None:
        """Compute the MP2 coefficients and energy corrections.

        See further up for more information.

        Args:
            grouped_property: A grouped second-quantized property that is required to contain the
                :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`.
                From this we require the two-body molecular orbitals electronic integrals and orbital
                energies. Optionally, it also obtain the Hartree-Fock reference energy to compute
                the absolute MP2 energy.
            ansatz: The UCC ansatz. Required to set the :attr:`excitation_list` to ensure that the
                coefficients are mapped correctly in the initial point array.

        Raises:
            QiskitNatureError: If :attr:`grouped_property` and/or :attr`ansatz` are not set.
        """
        missing_input_error_message: str = (
            "Not enough information has been provided to compute the MP2 corrections. "
            "Set the grouped property and the ansatz or call compute with them as arguments. "
            "The ansatz is not required if the excitation list has been set directly."
        )

        if isinstance(grouped_property, GroupedSecondQuantizedProperty):
            self.grouped_property = grouped_property
        elif not isinstance(self._grouped_property, GroupedSecondQuantizedProperty):
            raise QiskitNatureError(
                "The grouped property has not been set. " + missing_input_error_message
            )

        if isinstance(ansatz, UCC):
            self.ansatz = ansatz
        elif not isinstance(self._excitation_list, list):
            raise QiskitNatureError(
                "The excitation list has not been set directly or via the ansatz. "
                + missing_input_error_message
            )

        self._corrections = self._compute_corrections()

    def _compute_corrections(
        self,
    ) -> list[_Correction]:
        """Compute the MP2 coefficients and energy corrections.

        Non-double excitations will have zero coefficient and energy_correction.

        Returns:
            Dictionary with MP2 coefficients and energy_corrections for each excitation.
        """
        corrections: list[_Correction] = []
        for excitation in self._excitation_list:
            if len(excitation[0]) == 2:
                # Compute MP2 corrections using double excitations.
                corrections.append(self._compute_correction(excitation))
            else:
                # No computation for single, triple, and higher excitations.
                corrections.append(_Correction(excitation=excitation))

        return corrections

    def _compute_correction(
        self, excitation: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> _Correction:
        r"""Compute the MP2 coefficient and energy correction given a double excitation.

        Each double excitation indexed by :math:`i,a,j,b` has a correction coefficient,

        ..math::

            c_{i,a,j,b} = -\frac{2 T_{i,a,j,b} - T_{i,b,j,a}}{E_b + E_a - E_i - E_j},

        where :math:`E` is the orbital energy and :math:`T` is the integral matrix.
        And an energy correction,

        ..math::

            \Delta E_{i, a, j, b} = c_{i,a,j,b} T_{i,a,j,b}.

        These computations use molecular orbitals, but the indices used in the excitation
        information passed in and out use the block spin orbital numbering common to Qiskit Nature:
          - :math:`\alpha` runs from `0` to `num_molecular_orbitals - 1`,
          - :math:`\beta` runs from `num_molecular_orbitals` to
            `num_molecular_orbitals * 2 - 1`.

        Args:
            excitations: List of excitations.

        Returns:
            List of corrections.
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
        coefficient = -expectation_value / orbital_energy_correction
        coefficient = coefficient if abs(coefficient) > threshold else 0.0

        energy_correction = coefficient * expectation_value_iajb
        energy_correction = energy_correction if abs(energy_correction) > threshold else 0.0

        return _Correction(excitation=excitation, coefficient=coefficient, energy=energy_correction)

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as an array."""
        if self._corrections is None:
            self.compute()
        return np.asarray([correction.coefficient for correction in self._corrections])

    def get_energy_corrections(self) -> np.ndarray:
        """The energy corrections for each excitation."""
        if self._corrections is None:
            self.compute()
        return np.asarray([correction.energy for correction in self._corrections])

    def get_energy_correction(self) -> float:
        """The overall energy correction."""
        return self.get_energy_corrections().sum()

    def get_energy(self) -> float:
        """The absolute energy.

        If the reference energy was not obtained from
        :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`
        this will be equal to :meth:`get_energy_correction`.
        """
        return self._reference_energy + self.get_energy_correction()
