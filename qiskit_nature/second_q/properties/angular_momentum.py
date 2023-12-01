# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The AngularMomentum property."""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import FermionicOp

from .s_operators import s_minus_operator, s_plus_operator, s_z_operator

LOGGER = logging.getLogger(__name__)


class AngularMomentum:
    r"""The AngularMomentum property.

    The operator constructed by this property is the $S^2$ operator which is computed as:

    .. math::

       S^2 = (S^+ S^- + S^- S^+) / 2 + S^z S^z

    .. warning::

       If you are working with a non-orthogonal basis, you _must_ provide the ``overlap`` attribute
       in order to obtain the correct expectation value of this observable. Refer to the more
       extensive documentation of the :mod:`.s_operators` module for more details.

    See also:
        - the $S^z$ operator: :func:`.s_z_operator`
        - the $S^+$ operator: :func:`.s_plus_operator`
        - the $S^-$ operator: :func:`.s_minus_operator`

    The following attributes can be set via the initializer but can also be read and updated once
    the ``AngularMomentum`` object has been constructed.

    Attributes:
        num_spatial_orbitals (int): the number of spatial orbitals.
    """

    def __init__(self, num_spatial_orbitals: int, overlap: np.ndarray | None = None) -> None:
        r"""
        Args:
            num_spatial_orbitals: the number of spatial orbitals in the system.
            overlap: the overlap-matrix between the $\alpha$- and $\beta$-spin orbitals. When this
                is ``None``, the overlap-matrix is assumed to be identity.
        """
        self.num_spatial_orbitals = num_spatial_orbitals
        self._overlap: np.ndarray | None = None
        self.overlap = overlap

    @property
    def overlap(self) -> np.ndarray | None:
        r"""The overlap-matrix between the $\alpha$- and $\beta$-spin orbitals.

        When this is ``None``, the overlap-matrix is assumed to be identity.
        """
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: np.ndarray | None) -> None:
        self._overlap = overlap

        if overlap is not None:
            norb = self.num_spatial_orbitals
            delta = np.eye(2 * norb)
            delta[:norb, :norb] -= overlap.T @ overlap
            delta[norb:, norb:] -= overlap @ overlap.T
            summed = np.einsum("ij->", np.abs(delta))
            if not np.isclose(summed, 0.0, atol=1e-6):
                LOGGER.warning(
                    "The provided alpha-beta overlap matrix is NOT unitary! This can happen when "
                    "the alpha- and beta-spin orbitals do not span the same space. To provide an "
                    "example of what this means, consider an active space chosen from unrestricted-"
                    "spin orbitals. Computing <S^2> within this active space may not result in the "
                    "same <S^2> value as obtained on the single-reference starting point. More "
                    "importantly, this implies that the inactive subspace will account for the "
                    "difference between these two <S^2> values, possibly resulting in significant "
                    "spin contamination in both subspaces. You should verify whether this is "
                    "intentional/acceptable or whether your choice of active space can be improved."
                    " As a reference, here is the summed-absolute deviation of `S^T @ S` from the "
                    "identity: %s",
                    str(summed),
                )

    def second_q_ops(self) -> Mapping[str, FermionicOp]:
        """Returns the second quantized angular momentum operator.

        Returns:
            A mapping of strings to `FermionicOp` objects.
        """
        s_z = s_z_operator(self.num_spatial_orbitals)
        overlap_ab = self.overlap
        s_p = s_plus_operator(self.num_spatial_orbitals, overlap=overlap_ab)
        overlap_ba = overlap_ab.T if overlap_ab is not None else None
        s_m = s_minus_operator(self.num_spatial_orbitals, overlap=overlap_ba)

        spm_smp = (s_p @ s_m + s_m @ s_p).normal_order()
        op = 0.5 * spm_smp + s_z @ s_z

        return {self.__class__.__name__: op}

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.total_angular_momentum = []

        if result.aux_operators_evaluated is None:
            return

        for aux_op_eigenvalues in result.aux_operators_evaluated:
            if not isinstance(aux_op_eigenvalues, dict):
                continue

            _key = self.__class__.__name__

            if aux_op_eigenvalues[_key] is not None:
                result.total_angular_momentum.append(aux_op_eigenvalues[_key].real)
            else:
                result.total_angular_momentum.append(None)
