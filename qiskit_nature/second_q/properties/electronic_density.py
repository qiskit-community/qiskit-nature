# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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
from qiskit_nature.utils import get_einsum


class ElectronicDensity(ElectronicIntegrals):
    """The ElectronicDensity property.

    This property adds operators to evaluate the 1- and 2-reduced density matrices (RDMs). It is
    implemented as a subclass of the :class:`~qiskit_nature.second_q.operators.ElectronicIntegrals`
    which means that it supports mathematical operations and generally stores unrestricted spin
    data. However, you can trace out the spin using
    :meth:`~qiskit_nature.second_q.operators.ElectronicIntegrals.trace_spin`.
    """

    @staticmethod
    def _rdm2_from_rdm1s(
        rdm1_a: np.ndarray, rdm1_b: np.ndarray, *, mixed_spin: bool = False
    ) -> np.ndarray:
        einsum_func, _ = get_einsum()
        rdm2 = einsum_func("ij,kl->ijkl", rdm1_a, rdm1_b)
        if not mixed_spin:
            rdm2 -= einsum_func("ij,kl->ikjl", rdm1_a, rdm1_b)

        return rdm2

    @classmethod
    def empty(cls, num_spatial_orbitals: int, *, include_rdm2: bool = True) -> ElectronicDensity:
        """Initializes an empty (all-zero) ``ElectronicDensity``.

        Args:
            num_spatial_orbitals: the number of spatial orbitals.
            include_rdm2: whether to include the 2-body RDMs.

        Returns:
            The resulting ``ElectronicDensity``.
        """
        rdm1 = np.zeros((num_spatial_orbitals, num_spatial_orbitals), dtype=float)
        rdm2 = (
            np.zeros(
                (
                    num_spatial_orbitals,
                    num_spatial_orbitals,
                    num_spatial_orbitals,
                    num_spatial_orbitals,
                ),
                dtype=float,
            )
            if include_rdm2
            else None
        )

        return ElectronicDensity.from_raw_integrals(
            rdm1, rdm2, rdm1, rdm2, rdm2, auto_index_order=False
        )

    @classmethod
    def identity(cls, num_spatial_orbitals: int, *, include_rdm2: bool = True) -> ElectronicDensity:
        """Initializes an identity ``ElectronicDensity``.

        Args:
            num_spatial_orbitals: the number of spatial orbitals.
            include_rdm2: whether to include the 2-body RDMs.

        Returns:
            The resulting ``ElectronicDensity``.
        """
        rdm1 = np.eye(num_spatial_orbitals, dtype=float)
        rdm2: np.ndarray | None = None
        rdm2_ba: np.ndarray | None = None
        if include_rdm2:
            rdm2 = ElectronicDensity._rdm2_from_rdm1s(rdm1, rdm1)
            rdm2_ba = ElectronicDensity._rdm2_from_rdm1s(rdm1, rdm1, mixed_spin=True)

        return ElectronicDensity.from_raw_integrals(
            rdm1, rdm2, rdm1, rdm2, rdm2_ba, auto_index_order=False
        )

    @classmethod
    def from_particle_number(
        cls,
        num_spatial_orbitals: int,
        num_particles: int | tuple[int, int],
        *,
        include_rdm2: bool = True,
    ) -> ElectronicDensity:
        """Initializes an ``ElectronicDensity`` from the provided number of particles.

        Args:
            num_spatial_orbitals: the number of spatial orbitals.
            num_particles: the number of particles. If this is an integer it is interpreted as the
                total number of particles. If it is a tuple of two integers, these are treated as
                the number of alpha- and beta-spin particles, respectively.
            include_rdm2: whether to include the 2-body RDMs.

        Returns:
            The resulting ``ElectronicDensity``.
        """
        if isinstance(num_particles, int):
            num_beta = num_particles // 2
            num_alpha = num_particles - num_beta
        else:
            num_alpha, num_beta = num_particles

        rdm1_a = np.diag([1.0 if i < num_alpha else 0.0 for i in range(num_spatial_orbitals)])
        rdm1_b = np.diag([1.0 if i < num_beta else 0.0 for i in range(num_spatial_orbitals)])

        rdm2_aa: np.ndarray | None = None
        rdm2_bb: np.ndarray | None = None
        rdm2_ba: np.ndarray | None = None
        if include_rdm2:
            rdm2_aa = ElectronicDensity._rdm2_from_rdm1s(rdm1_a, rdm1_a)
            rdm2_bb = ElectronicDensity._rdm2_from_rdm1s(rdm1_b, rdm1_b)
            rdm2_ba = ElectronicDensity._rdm2_from_rdm1s(rdm1_b, rdm1_a, mixed_spin=True)

        return cls.from_raw_integrals(
            rdm1_a, rdm2_aa, rdm1_b, rdm2_bb, rdm2_ba, auto_index_order=False
        )

    @classmethod
    def from_orbital_occupation(
        cls,
        alpha_occupation: Sequence[float],
        beta_occupation: Sequence[float],
        *,
        include_rdm2: bool = True,
    ) -> ElectronicDensity:
        """Initializes an ``ElectronicDensity`` from the provided orbital occupations.

        This method assumes an orthonormal basis, and will initialize the 1-RDMs simply as diagonal
        matrices of the provided orbital occupations. The 2-RDMs are computed based on these 1-RDMs.

        Args:
            alpha_occupation: the alpha-spin orbital occupations.
            beta_occupation: the beta-spin orbital occupations.
            include_rdm2: whether to include the 2-body RDMs.

        Returns:
            The resulting ``ElectronicDensity``.
        """
        rdm1_a = np.diag(alpha_occupation)
        rdm1_b = np.diag(beta_occupation)

        rdm2_aa: np.ndarray | None = None
        rdm2_bb: np.ndarray | None = None
        rdm2_ba: np.ndarray | None = None
        if include_rdm2:
            rdm2_aa = ElectronicDensity._rdm2_from_rdm1s(rdm1_a, rdm1_a)
            rdm2_bb = ElectronicDensity._rdm2_from_rdm1s(rdm1_b, rdm1_b)
            rdm2_ba = ElectronicDensity._rdm2_from_rdm1s(rdm1_b, rdm1_a, mixed_spin=True)

        return cls.from_raw_integrals(
            rdm1_a, rdm2_aa, rdm1_b, rdm2_bb, rdm2_ba, auto_index_order=False
        )

    def second_q_ops(self) -> Mapping[str, FermionicOp]:
        """Returns the density evaluation operators.

        Returns:
            A mapping of strings to `FermionicOp` objects.
        """
        tensor = self.second_q_coeffs()
        register_length = tensor.register_length
        half = register_length // 2

        aux_ops = {}
        for key in tensor:
            if key == "":
                continue

            label_template = " ".join(f"{op}_{{}}" for op in key)

            ndarray = cast(np.ndarray, tensor[key])
            for index in np.ndindex(*ndarray.shape):
                if not _filter_index(index, half):
                    continue

                aux_ops[f"RDM{index}"] = FermionicOp(
                    {label_template.format(*index): 1.0},
                    num_spin_orbitals=register_length,
                    copy=False,
                )

        return aux_ops

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`qiskit_nature.second_q.problems.EigenstateResult`.

        In particular, this method gathers the evaluated auxiliary operator values and constructs
        the resulting ``ElectronicDensity`` and stores it in the result object.

        Args:
            result: the result to add meaning to.
        """
        n_spatial = self.register_length

        rdm1_idx_regex = re.compile(r"RDM\((\d+), (\d+)\)")
        rdm2_idx_regex = re.compile(r"RDM\((\d+), (\d+), (\d+), (\d+)\)")

        contains_rdm2 = "++--" in self.alpha.keys()

        rdm1_a = np.zeros((n_spatial, n_spatial), dtype=float)
        rdm1_b = np.zeros((n_spatial, n_spatial), dtype=float)

        rdm2_aa: np.ndarray | None = None
        rdm2_bb: np.ndarray | None = None
        rdm2_ba: np.ndarray | None = None
        if contains_rdm2:
            rdm2_aa = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=float)
            rdm2_bb = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=float)
            rdm2_ba = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=float)

        for name, aux_value in result.aux_operators_evaluated[0].items():
            # immediately skip zero values
            if np.isclose(aux_value, 0.0):
                continue

            match = rdm1_idx_regex.fullmatch(name)
            if match is not None:
                index = tuple(int(idx) for idx in match.groups())
                bools = _boolean_index(index, n_spatial)

                if all(bools):
                    rdm1_a[index] = aux_value.real
                elif not any(bools):
                    rdm1_b[_shifted_index(index, n_spatial)] = aux_value.real

                continue

            if contains_rdm2:
                match = rdm2_idx_regex.fullmatch(name)
                if match is not None:
                    index = tuple(int(idx) for idx in match.groups())
                    bools = _boolean_index(index, n_spatial)

                    if all(bools):
                        rdm2_aa[index] = aux_value.real
                    elif not any(bools):
                        rdm2_bb[_shifted_index(index, n_spatial)] = aux_value.real
                    elif bools[0] and bools[3] and not bools[1] and not bools[2]:
                        rdm2_ba[_shifted_index(index, n_spatial)] = aux_value.real

        result.electronic_density = ElectronicDensity.from_raw_integrals(
            rdm1_a, rdm2_aa, rdm1_b, rdm2_bb, rdm2_ba, auto_index_order=False
        )


def _boolean_index(index: tuple[int, ...], size: int) -> tuple[bool, ...]:
    return tuple(i < size for i in index)


def _shifted_index(index: tuple[int, ...], size: int) -> tuple[int, ...]:
    bools = _boolean_index(index, size)
    return tuple(idx if bools[i] else idx - size for i, idx in enumerate(index))


def _filter_index(index: tuple[int, ...], size: int) -> bool:
    bools = _boolean_index(index, size)
    nbody = len(index)

    if nbody == 2:
        return all(bools) or not any(bools)

    if nbody == 4:
        return (
            all(bools)  # alpha-alpha
            or not any(bools)  # beta-beta
            or (bools[0] and bools[3] and not bools[1] and not bools[2])  # beta-alpha
        )
    return False
