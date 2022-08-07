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
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.properties import ElectronicEnergy, GroupedSecondQuantizedProperty
from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.properties.integrals import ElectronicIntegrals


from .initial_point import InitialPoint


class MP2InitialPoint(InitialPoint):
    """Compute the second-order Møller-Plesset perturbation theory (MP2) initial point.

    The computed MP2 correction coefficients are intended for use as an initial point for the VQE
    parameters in combination with a
    :class:`~qiskit_nature.second_q.circuit.library.ansatzes.ucc.UCC` ansatz.

    The coefficients and energy corrections are computed using the :meth:`compute` method, which
    requires the :attr:`grouped_property` and :attr:`ansatz` to be passed as arguments or the
    :attr:`grouped_property` and :attr:`excitation_list` attributes to be set already.

    ``MP2InitialPoint`` requires the :class:`~qiskit_nature.second_q.properties.ElectronicEnergy`,
    which should be passed in via the :attr:`grouped_property` attribute. From this it must obtain
    the two-body molecular orbital electronic integrals and orbital energies. If the Hartree-Fock
    reference energy is also obtained, it will be used to compute the absolute MP2 energy using the
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
        self._ansatz: UCC | None = None
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None
        self.threshold: float = threshold
        self._integral_matrix: np.ndarray | None = None
        self._orbital_energies: np.ndarray | None = None
        self._reference_energy: float = 0.0

        # T amplitudes t2[i,j,a,b]  (i,j in occ, a,b in vir)
        self._t2_amplitudes: np.ndarray | None = None
        self._energy_correction: float = 0.0
        self._amplitudes: np.ndarray | None = None

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz.

        This is used to ensure that the :attr:`excitation_list` matches with the UCC ansatz that
        will be used with the VQE algorithm.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:

        # Operators must be built early to compute the excitation list.
        _ = ansatz.operators

        self._invalidate()

        self._excitation_list = ansatz.excitation_list
        self._ansatz = ansatz

    @property
    def excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The list of excitations.

        Setting this will overwrite the excitation list from the ansatz.
        """
        return self._excitation_list

    @excitation_list.setter
    def excitation_list(self, excitations: list[tuple[tuple[int, ...], tuple[int, ...]]]):
        self._invalidate()

        self._excitation_list = excitations

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty | None:
        """The grouped property.

        The grouped property is required to contain the
        :class:`~qiskit_nature.second_q.properties.ElectronicEnergy`, which must contain the
        two-body molecular orbitals matrix and the orbital energies. Optionally, it will also use
        the Hartree-Fock reference energy to compute the absolute energy.

        Raises:
            QiskitNatureError: If :class:`~qiskit_nature.second_q.properties.ElectronicEnergy`
                is missing or the two-body molecular orbitals matrix or the orbital energies are not
                found.
            NotImplementedError: If alpha and beta spin molecular orbitals are not identical.
        """
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:

        electronic_energy: ElectronicEnergy | None = grouped_property.get_property(ElectronicEnergy)
        if electronic_energy is None:
            raise QiskitNatureError(
                "The ElectronicEnergy cannot be obtained from the grouped_property."
            )

        two_body_mo_integral: ElectronicIntegrals | None = (
            electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
        )
        if two_body_mo_integral is None:
            raise QiskitNatureError(
                "The two body MO electronic integral cannot be obtained from the grouped property."
            )

        orbital_energies: np.ndarray | None = electronic_energy.orbital_energies
        if orbital_energies is None:
            raise QiskitNatureError(
                "The orbital_energies cannot be obtained from the grouped property."
            )

        integral_matrix: np.ndarray = two_body_mo_integral.get_matrix()
        if not np.allclose(integral_matrix, two_body_mo_integral.get_matrix(2)):
            raise NotImplementedError(
                "MP2InitialPoint only supports restricted-spin setups. "
                "Alpha and beta spin orbitals must be identical. "
                "See https://github.com/Qiskit/qiskit-nature/issues/645."
            )

        self._invalidate()

        num_occ = len(orbital_energies[orbital_energies < 0.0])

        # Use NumPy broadcasting to compute all occupied-virtual energy deltas.
        energy_deltas = (
            orbital_energies[:num_occ, np.newaxis] - orbital_energies[np.newaxis, num_occ:]
        )
        double_deltas = energy_deltas[:, :, np.newaxis, np.newaxis] + energy_deltas

        # Create integral matrix that uses occupied and virtual indices rather than MO indices.
        integral_matrix_ovov = integral_matrix[:num_occ, num_occ:, :num_occ, num_occ:]

        # Compute T2 amplitudes and transpose to num_occ, num_occ, num_vir, num_vir.
        t2_amplitudes = (integral_matrix_ovov / double_deltas).transpose(0, 2, 1, 3)

        # Compute MP2 energy correction.
        energy_correction = np.einsum("ijab,iajb", t2_amplitudes, integral_matrix_ovov) * 2
        energy_correction -= np.einsum("ijab,ibja", t2_amplitudes, integral_matrix_ovov)

        self._t2_amplitudes = t2_amplitudes
        self._energy_correction = energy_correction
        self._orbital_energies = orbital_energies
        self._integral_matrix = integral_matrix
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

        self._invalidate()

        self._threshold = threshold

    def compute(
        self,
        ansatz: UCC | None = None,
        grouped_property: GroupedSecondQuantizedProperty | None = None,
    ) -> None:
        """Compute the coefficients and energy corrections.

        See further up for more information.

        Args:
            grouped_property: A grouped second-quantized property that may optionally contain the
                Hartree-Fock reference energy. This is for consistency with other initial points.
            ansatz: The UCC ansatz. Required to set the :attr:`excitation_list` to ensure that the
                coefficients are mapped correctly in the initial point array.

        Raises:
            QiskitNatureError: If :attr:`_excitation_list` or :attr:`_grouped_property` is not set.
        """
        if ansatz is not None:
            self.ansatz = ansatz

        if self._excitation_list is None:
            raise QiskitNatureError(
                "The excitation list has not been set directly or via the ansatz. "
                "Not enough information has been provided to compute the initial point. "
                "Set the ansatz or call compute with it as an argument. "
                "The ansatz is not required if the excitation list has been set directly."
            )

        if grouped_property is not None:
            self.grouped_property = grouped_property

        if self._grouped_property is None:
            raise QiskitNatureError("The grouped_property has not been set.")

        self._compute()

    def _compute(self) -> None:
        """Compute the MP2 coefficients and energy corrections.

        Non-double excitations will have zero coefficient and energy_correction.

        Returns:
            Dictionary with MP2 coefficients and energy_corrections for each excitation.
        """
        num_occ = self._t2_amplitudes.shape[0]
        amplitudes = np.zeros(len(self.excitation_list))
        for index, excitation in enumerate(self._excitation_list):
            if len(excitation[0]) == 2:
                # Get the amplitude of the double excitation.
                [[i, j], [a, b]] = np.asarray(excitation) % num_occ
                amplitude = self._t2_amplitudes[i, j, a - num_occ, b - num_occ]
                amplitudes[index] = amplitude if abs(amplitude) > self._threshold else 0.0

        self._amplitudes = amplitudes

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as an array."""
        if self._amplitudes is None:
            self.compute()
        return self._amplitudes

    def get_energy_correction(self) -> float:
        """The overall energy correction."""
        return self._energy_correction

    def get_energy(self) -> float:
        """The absolute energy.

        If the reference energy was not obtained from
        :class:`~qiskit_nature.second_q.properties.ElectronicEnergy` this will be equal to
        :meth:`get_energy_correction`.
        """
        return self._reference_energy + self.get_energy_correction()

    def _invalidate(self):
        """Invalidate any previous computation."""
        self._amplitudes = None
