# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ElectronicDensity property."""

from __future__ import annotations

import re
from typing import Mapping, Sequence, cast

import numpy as np

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp


class ElectronicDensity(ElectronicIntegrals):
    """The ElectronicDensity property.

    This property adds operators to evaluate the 1- and 2-reduced density matrices (RDMs). It is
    implemented as a subclass of the :class:`~qiskit_nature.second_q.operators.ElectronicIntegrals`
    which means that it supports mathematical operations and generally stores unrestricted spin
    data. However, you can trace out the spin using
    :meth:`~qiskit_nature.second_q.operators.ElectronicIntegrals.trace_spin`.
    """

    @staticmethod
    def from_orbital_occupation(
        alpha_occupation: Sequence[float],
        beta_occupation: Sequence[float],
    ) -> ElectronicDensity:
        """Initializes an ``ElectronicDensity`` from the provided orbital occupations.

        This method assumes an orthonormal basis, and will initialize the 1-RDMs simply as diagonal
        matrices of the provided orbital occupations. The 2-RDMs are computed based on these 1-RDMs.

        Args:
            alpha_occupation: the alpha-spin orbital occupations.
            beta_occupation: the beta-spin orbital occupations.

        Returns:
            The resulting ``ElectronicDensity``.
        """
        rdm1_a = np.diag(alpha_occupation)
        rdm2_aa = np.einsum("ij,kl->ijkl", rdm1_a, rdm1_a) - np.einsum(
            "ij,kl->iklj", rdm1_a, rdm1_a
        )

        rdm1_b = np.diag(beta_occupation)
        rdm2_bb = np.einsum("ij,kl->ijkl", rdm1_b, rdm1_b) - np.einsum(
            "ij,kl->iklj", rdm1_b, rdm1_b
        )
        rdm2_ba = np.einsum("ij,kl->ijkl", rdm1_a, rdm1_b)

        return ElectronicDensity.from_raw_integrals(
            rdm1_a, rdm2_aa, rdm1_b, rdm2_bb, rdm2_ba, auto_index_order=False
        )

    def second_q_ops(self) -> Mapping[str, FermionicOp]:
        """Returns the density evaluation operators.

        Returns:
            A mapping of strings to `FermionicOp` objects.
        """
        tensor = self.second_q_coeffs()
        register_length = tensor.register_length

        aux_ops = {}
        for key in tensor:
            if key == "":
                continue

            label_template = " ".join(f"{op}_{{}}" for op in key)

            ndarray = cast(np.ndarray, tensor[key])
            for index in np.ndindex(*ndarray.shape):
                aux_ops[f"RDM{index}"] = FermionicOp(
                    {label_template.format(*index): 1.0},
                    num_spin_orbitals=register_length,
                    copy=False,
                )

        return aux_ops

    def interpret(
        self, result: "qiskit_nature.second_q.problemsEigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`qiskit_nature.second_q.problems.EigenstateResult`.

        In particular, this method gathers the evaluated auxiliary operator values and constructs
        the resulting ``ElectronicDensity`` and stores it in the result object.

        Args:
            result: the result to add meaning to.
        """
        n_spatial = self.register_length
        n_spin = 2 * n_spatial

        rdm1 = np.zeros((n_spin, n_spin), dtype=float)
        rdm2 = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=float)

        rdm1_idx_regex = re.compile(r"RDM\((\d+), (\d+)\)")
        rdm2_idx_regex = re.compile(r"RDM\((\d+), (\d+), (\d+), (\d+)\)")

        for name, aux_value in result.aux_operators_evaluated[0].items():
            # immediately skip zero values
            if np.isclose(aux_value, 0.0):
                continue

            match = rdm1_idx_regex.fullmatch(name)
            if match is not None:
                mo_i, mo_j = (int(idx) for idx in match.groups())
                rdm1[mo_i, mo_j] = aux_value.real
                continue

            match = rdm2_idx_regex.fullmatch(name)
            if match is not None:
                mo_i, mo_j, mo_k, mo_l = (int(idx) for idx in match.groups())
                rdm2[mo_i, mo_j, mo_k, mo_l] = aux_value.real

        result.electronic_density = ElectronicDensity.from_raw_integrals(
            rdm1[n_spatial:, n_spatial:],
            rdm2[n_spatial:, n_spatial:, n_spatial:, n_spatial:],
            rdm1[:n_spatial, :n_spatial],
            rdm2[:n_spatial, :n_spatial, :n_spatial, :n_spatial],
            rdm2[:n_spatial, n_spatial:, n_spatial:, :n_spatial],
            auto_index_order=False,
        )
