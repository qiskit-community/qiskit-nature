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

"""The orbital rotation class"""

from typing import Tuple
import logging

import numpy as np
from scipy.linalg import expm

from qiskit_nature import QiskitNatureError
from qiskit_nature.converters.second_quantization import QubitConverter

logger = logging.getLogger(__name__)

class OrbitalRotation:
    """Class that regroups methods for creation of matrices that rotate the MOs.
    It allows to create the unitary matrix U = exp(-kappa) that is parameterized with kappa's
    elements. The parameters are the off-diagonal elements of the anti-hermitian matrix kappa.
    """

    def __init__(
        self,
        num_qubits: int,
        qubit_converter: QubitConverter,
        orbital_rotations: list = None,
        orbital_rotations_beta: list = None,
        parameters: list = None,
        parameter_bounds: list = None,
        parameter_initial_value: float = 0.1,
        parameter_bound_value: Tuple[float, float] = (-2 * np.pi, 2 * np.pi),
    ) -> None:
        """
        Args:
            num_qubits: number of qubits necessary to simulate a particular system.
            transformation: a fermionic driver to operator transformation strategy.
            qmolecule: instance of the :class:`~qiskit_nature.drivers.QMolecule`
                class which has methods
                needed to recompute one-/two-electron/dipole integrals after orbital rotation
                (C = C0 * exp(-kappa)). It is not required but can be used if user wished to
                provide custom integrals for instance.
            orbital_rotations: list of alpha orbitals that are rotated (i.e. [[0,1], ...] the
                0-th orbital is rotated with 1-st, which corresponds to non-zero entry 01 of
                the matrix kappa).
            orbital_rotations_beta: list of beta orbitals that are rotated.
            parameters: orbital rotation parameter list of matrix elements that rotate the MOs,
                each associated to a pair of orbitals that are rotated
                (non-zero elements in matrix kappa), or elements in the orbital_rotation(_beta)
                lists.
            parameter_bounds: parameter bounds
            parameter_initial_value: initial value for all the parameters.
            parameter_bound_value: value for the bounds on all the parameters
        """

        self._num_qubits = num_qubits
        self._qubit_converter = qubit_converter

        self._orbital_rotations = orbital_rotations
        self._orbital_rotations_beta = orbital_rotations_beta
        self._parameter_initial_value = parameter_initial_value
        self._parameter_bound_value = parameter_bound_value
        self._parameters = parameters
        if self._parameters is None:
            self._create_parameter_list_for_orbital_rotations()

        self._num_parameters = len(self._parameters)
        self._parameter_bounds = parameter_bounds
        if self._parameter_bounds is None:
            self._create_parameter_bounds()

        if self._qubit_converter.two_qubit_reduction is True:
            self._dim_kappa_matrix = int((self._num_qubits + 2) / 2)
        else:
            self._dim_kappa_matrix = int(self._num_qubits / 2)

        self._check_for_errors()
        self._matrix_a = None
        self._matrix_b = None

    def _check_for_errors(self) -> None:
        """Checks for errors such as incorrect number of parameters and indices of orbitals."""

        # number of parameters check
        if self._orbital_rotations_beta is None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) != len(self._parameters):
                raise QiskitNatureError(
                    f"Please specify same number of params ({len(self._parameters)}) as there are "
                    f"orbital rotations ({len(self._orbital_rotations)})"
                )
        elif self._orbital_rotations_beta is not None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) + len(self._orbital_rotations_beta) != len(
                self._parameters
            ):
                raise QiskitNatureError(
                    f"Please specify same number of params ({len(self._parameters)}) as there are "
                    f"orbital rotations ({len(self._orbital_rotations)})"
                )
        # indices of rotated orbitals check
        for exc in self._orbital_rotations:
            if exc[0] > (self._dim_kappa_matrix - 1):
                raise QiskitNatureError(
                    f"You specified entries that go outside "
                    f"the orbital rotation matrix dimensions {exc[0]}, "
                )
            if exc[1] > (self._dim_kappa_matrix - 1):
                raise QiskitNatureError(
                    f"You specified entries that go outside "
                    f"the orbital rotation matrix dimensions {exc[1]}"
                )
        if self._orbital_rotations_beta is not None:
            for exc in self._orbital_rotations_beta:
                if exc[0] > (self._dim_kappa_matrix - 1):
                    raise QiskitNatureError(
                        f"You specified entries that go outside "
                        f"the orbital rotation matrix dimensions {exc[0]}"
                    )
                if exc[1] > (self._dim_kappa_matrix - 1):
                    raise QiskitNatureError(
                        f"You specified entries that go outside "
                        f"the orbital rotation matrix dimensions {exc[1]}"
                    )

    def _create_orbital_rotation_list(self) -> None:
        """Creates a list of indices of matrix kappa that denote the pairs of orbitals that
        will be rotated. For instance, a list of pairs of orbital such as [[0,1], [0,2]]."""

        if self._qubit_converter.two_qubit_reduction:
            half_as = int((self._num_qubits + 2) / 2)
        else:
            half_as = int(self._num_qubits / 2)

        self._orbital_rotations = []
        for i in range(half_as):
            for j in range(half_as):
                if i < j:
                    self._orbital_rotations.append([i, j])

    def _create_parameter_list_for_orbital_rotations(self) -> None:
        """Initializes the initial values of orbital rotation matrix kappa."""

        # creates the indices of matrix kappa and prevent user from trying to rotate only betas
        if self._orbital_rotations is None:
            self._create_orbital_rotation_list()
        elif self._orbital_rotations is None and self._orbital_rotations_beta is not None:
            raise QiskitNatureError(
                "Only beta orbitals labels (orbital_rotations_beta) have been provided."
                "Please also specify the alpha orbitals (orbital_rotations) "
                "that are rotated as well. Do not specify anything to have by default "
                "all orbitals rotated."
            )

        if self._orbital_rotations_beta is not None:
            num_parameters = len(self._orbital_rotations + self._orbital_rotations_beta)
        else:
            num_parameters = len(self._orbital_rotations)
        self._parameters = [self._parameter_initial_value for _ in range(num_parameters)]

    def _create_parameter_bounds(self) -> None:
        """Create bounds for parameters."""
        self._parameter_bounds = [self._parameter_bound_value for _ in range(self._num_parameters)]

    def get_orbital_rotation_matrix(self, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates 2 matrices K_alpha, K_beta that rotate the orbitals through MO coefficient
        C_alpha = C_RHF * U_alpha where U = e^(K_alpha), similarly for beta orbitals."""

        self._parameters = parameters  # type: ignore
        k_matrix_alpha = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))
        k_matrix_beta = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))
        # allows to selectively rotate pairs of orbitals
        if self._orbital_rotations_beta is None:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[i]
        else:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]

            for j, exc in enumerate(self._orbital_rotations_beta):
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[j + len(self._orbital_rotations)]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[j + len(self._orbital_rotations)]

        self._matrix_a = expm(k_matrix_alpha)
        self._matrix_b = expm(k_matrix_beta)

        return self._matrix_a, self._matrix_b

    @property
    def matrix_a(self) -> np.ndarray:
        """Returns matrix A."""
        return self._matrix_a

    @property
    def matrix_b(self) -> np.ndarray:
        """Returns matrix B."""
        return self._matrix_b

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters."""
        return self._num_parameters

    @property
    def parameter_bound_value(self) -> Tuple[float, float]:
        """Returns a value for the bounds on all the parameters."""
        return self._parameter_bound_value
