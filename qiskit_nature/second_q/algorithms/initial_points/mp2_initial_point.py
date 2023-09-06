# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
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
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.operators.tensor import Tensor
from qiskit_nature.second_q.operators.symmetric_two_body import SymmetricTwoBodyIntegrals, unfold
from qiskit_nature.second_q.problems import BaseProblem, ElectronicStructureProblem
from qiskit_nature.utils import get_einsum

from .initial_point import InitialPoint


def _compute_mp2(
    num_occ: int, integral_matrix: Tensor, orbital_energies: np.ndarray
) -> tuple[np.ndarray, float]:
    """Compute the T2 amplitudes and MP2 energy correction.

    Args:
        num_occ: The number of occupied molecular orbitals.
        integral_matrix: The two-body molecular orbitals tensor.
        orbital_energies: The orbital energies.

    Returns:
        A tuple consisting of the:
        - T amplitudes t2[i, j, a, b] (i, j in occupied, a, b in virtual).
        - The MP2 energy correction.

    """
    if isinstance(integral_matrix, SymmetricTwoBodyIntegrals):
        integral_matrix = unfold(integral_matrix)

    # We use NumPy broadcasting to compute the matrix of occupied - virtual energy deltas with
    # shape (num_occ, num_vir), such that
    # energy_deltas[i, a] = orbital_energy[i] - orbital_energy[a].
    # NOTE: In the unrestricted-spin calculation, the orbital energies will be a 2D array, and this
    # logic will need to be revisited.
    energy_deltas = orbital_energies[:num_occ, np.newaxis] - orbital_energies[num_occ:]

    # We now want to compute a 4D tensor of (occupied, occupied) - (virtual, virtual)
    # energy deltas with shape (num_occ, num_vir, num_occ, num_vir), such that
    # double_deltas[i, a, j, b] = orbital_energies[i] + orbital_energies[j]
    #                             - orbital_energies[a] - orbital_energies[b].
    # Again we can use NumPy broadcasting to speed this up.
    double_deltas = energy_deltas[:, :, np.newaxis, np.newaxis] + energy_deltas

    # Now we transpose this matrix into (num_occ, num_occ, num_vir, num_vir) which is the expected
    # ordering of T2 amplitudes following the convention set out by PySCF
    double_deltas = double_deltas.transpose(0, 2, 1, 3)

    # We must swap the last two axes in order to match the ordering of double_deltas as
    # explained above. We use the _reverse_label_template routine here to handle non-default label
    # templates in the integral_matrix Tensor, too.
    integral_matrix = integral_matrix.transpose(
        integral_matrix._reverse_label_template((0, 1, 3, 2))
    )
    # Create integral matrix that uses occupied and virtual indices rather than MO indices.
    integral_matrix_oovv = integral_matrix[:num_occ, :num_occ, num_occ:, num_occ:]

    # Compute T2 amplitudes and transpose to num_occ, num_occ, num_vir, num_vir.
    t2_amplitudes = integral_matrix_oovv / double_deltas

    # Compute MP2 energy correction.
    einsum_func, _ = get_einsum()
    energy_correction = einsum_func("ijab,ijab", t2_amplitudes, integral_matrix_oovv) * 2
    energy_correction -= einsum_func("ijab,ijba", t2_amplitudes, integral_matrix_oovv)

    return t2_amplitudes, energy_correction


class MP2InitialPoint(InitialPoint):
    """Compute the second-order Møller-Plesset perturbation theory (MP2) initial point.

    The computed MP2 correction coefficients are intended for use as an initial point for the VQE
    parameters in combination with a
    :class:`~qiskit_nature.second_q.circuit.library.ansatzes.ucc.UCC` ansatz.

    The initial point parameters are computed using the :meth:`compute` method, which requires the
    :attr:`problem` and :attr:`ansatz` to be passed as arguments or the
    :attr:`problem` and :attr:`ansatz` attributes to be set already.

    The :attr:`problem` is required to be an
    :class:`~qiskit_nature.second_q.problems.ElectronicStructureProblem` which contains an
    :class:`~qiskit_nature.second_q.hamiltonians.ElectronicEnergy` Hamiltonian.
    It also must have its ``num_particles`` and ``orbital_energies`` attributes specified. If its
    ``reference_energy`` attribute is provided, this will be used to compute the
    :attr:`total_energy`.
    :class:`~qiskit_nature.second_q.hamiltonians.ElectronicEnergy` must contain
    the two-body, molecular-orbital ``electronic_integrals``.

    Setting the :attr:`problem` will compute the :attr:`t2_amplitudes` and
    :attr:`energy_correction`.

    Following computation, one can obtain the initial point array via the :meth:`to_numpy_array`
    method. The initial point parameters that correspond to double excitations in the
    ``excitation_list`` will equal the appropriate T2 amplitude, while those below
    :attr:`threshold` or that correspond to single, triple, or higher excitations will be zero.
    """

    def __init__(self, *, threshold: float = 1e-12) -> None:
        super().__init__()
        self.threshold: float = threshold
        self._ansatz: UCC | None = None
        self._t2_amplitudes: np.ndarray | None = None
        self._parameters: np.ndarray | None = None
        self._energy_correction: float = 0.0
        self._total_energy: float = 0.0

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz.

        The ``excitation_list`` and ``reps`` used by the
        :class:`~qiskit_nature.circuit.library.ansatzes.ucc.UCC` ansatz is obtained to ensure that
        the shape of the initial point is appropriate.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        self._invalidate()
        self._ansatz = ansatz

    @property
    def problem(self) -> BaseProblem | None:
        """The problem instance.

        See :class:`~qiskit_nature.second_q.algorithms.initial_points.MP2InitialPoint` for more
        information on the required properties.

        Raises:
            QiskitNatureError: If :class:`~qiskit_nature.second_q.hamiltonians.ElectronicEnergy` is
                missing or the two-body molecular orbitals matrix.
            QiskitNatureError: If
                :attr:`~qiskit_nature.second_q.problems.ElectronicStructureProblem.num_particles` or
                :attr:`~qiskit_nature.second_q.problems.ElectronicStructureProblem.orbital_energies`
                is ``None``.
            NotImplementedError: If alpha and beta spin molecular orbitals are not identical.
        """
        return self._problem

    @problem.setter
    def problem(self, problem: BaseProblem) -> None:
        if not isinstance(problem, ElectronicStructureProblem):
            raise QiskitNatureError(
                "Only an `ElectronicStructureProblem` is compatible with the MP2InitialPoint, not a"
                f" problem of type, {type(problem)}."
            )

        electronic_energy = problem.hamiltonian
        if electronic_energy is None:
            raise QiskitNatureError("The `ElectronicEnergy` cannot be obtained from the `problem`.")

        two_body_mo_integral: ElectronicIntegrals = electronic_energy.electronic_integrals.two_body
        if two_body_mo_integral.alpha.is_empty():
            raise QiskitNatureError(
                "The alpha-alpha spin two-body MO `electronic_integrals` cannot be empty."
            )

        orbital_energies: np.ndarray | None = problem.orbital_energies
        if orbital_energies is None:
            raise QiskitNatureError("The `orbital_energies` cannot be obtained from the `problem`.")

        if two_body_mo_integral.beta.get("++--", None) is not None:
            raise NotImplementedError(
                "`MP2InitialPoint` only supports restricted-spin setups. "
                "Alpha and beta spin orbitals must be identical. "
                "See https://github.com/Qiskit/qiskit-nature/issues/645."
            )

        # Get number of occupied molecular orbitals as the number of alpha particles.
        # Only valid for restricted-spin setups.
        num_occ = problem.num_alpha
        if num_occ is None:
            raise QiskitNatureError(
                "The `num_particles` attribute of the `ElectronicStructureProblem` is required by "
                "the MP2InitialPoint."
            )

        integral_matrix = two_body_mo_integral.alpha.get("++--")

        reference_energy = problem.reference_energy if not None else 0.0

        self._invalidate()

        t2_amplitudes, energy_correction = _compute_mp2(num_occ, integral_matrix, orbital_energies)

        # Save state.
        self._problem = problem
        self._t2_amplitudes = t2_amplitudes
        self._energy_correction = energy_correction
        self._total_energy = reference_energy + energy_correction

    @property
    def t2_amplitudes(self) -> np.ndarray:
        """T2 amplitudes.

        Given ``t2[i, j, a, b]`` ``i, j`` carry virtual indices, while ``a, b`` carry occupied
        indices.
        """
        return self._t2_amplitudes

    @property
    def energy_correction(self) -> float:
        """The MP2 energy correction."""
        return self._energy_correction

    @property
    def total_energy(self) -> float:
        """The total energy including the Hartree-Fock energy.

        If the reference energy was not obtained from
        :class:`~qiskit_nature.second_q.hamiltonians.ElectronicEnergy` this will be equal to
        :attr:`energy_correction`.
        """
        return self._total_energy

    @property
    def threshold(self) -> float:
        """Amplitudes below this vanish in the initial point array."""
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
        problem: BaseProblem | None = None,
    ) -> None:
        """Compute the initial point parameter for each excitation.

        See class documentation for more information.

        Args:
            problem: The :attr:`problem`.
            ansatz: The :attr:`ansatz`.

        Raises:
            QiskitNatureError: If :attr:`ansatz` or :attr:`problem` is not set.
        """
        if ansatz is not None:
            self.ansatz = ansatz

        if self._ansatz is None:
            raise QiskitNatureError(
                "The ansatz property has not been set. "
                "Not enough information has been provided to compute the initial point. "
                "Set the ansatz or call compute with it as an argument. "
            )

        if problem is not None:
            self.problem = problem

        if self._problem is None:
            raise QiskitNatureError("The problem has not been set.")

        self._compute()

    def _compute(self) -> None:
        """Compute the MP2 amplitudes given an excitation list.

        Non-double excitations will have zero coefficient.

        Returns:
            The MP2 T2 amplitudes for each excitation.
        """
        # Operators must be built to compute the excitation list.
        _ = self._ansatz.operators
        num_occ = self._t2_amplitudes.shape[0]
        amplitudes = np.zeros(len(self._ansatz.excitation_list), dtype=float)
        for index, excitation in enumerate(self._ansatz.excitation_list):
            if len(excitation[0]) == 2:
                # Get the amplitude of the double excitation.
                [[i, j], [a, b]] = np.asarray(excitation) % num_occ
                amplitude = self._t2_amplitudes[i, j, a - num_occ, b - num_occ]
                amplitudes[index] = amplitude if abs(amplitude) > self._threshold else 0.0

        self._parameters = np.tile(amplitudes, self._ansatz.reps)

    def to_numpy_array(self) -> np.ndarray:
        """The initial point as a NumPy array."""
        if self._parameters is None:
            self.compute()
        return self._parameters

    def _invalidate(self):
        """Invalidate any previous computation."""
        self._parameters = None
