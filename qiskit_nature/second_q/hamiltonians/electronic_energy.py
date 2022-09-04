# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ElectronicEnergy property."""

from __future__ import annotations

from typing import Optional, cast, TYPE_CHECKING

import numpy as np

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.properties.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.second_q.properties.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from .hamiltonian import Hamiltonian

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import EigenstateResult


class ElectronicEnergy(Hamiltonian):
    """The ElectronicEnergy property.

    This is the main property of any electronic structure problem. It constructs the Hamiltonian
    whose eigenvalue is the target of a later used Quantum algorithm.

    Note that this Property computes **purely** the electronic energy (possibly minus additional
    shifts introduced via e.g. classical transformers). However, for convenience it provides a
    storage location for the nuclear repulsion energy. If available, this information will be used
    during the call of ``interpret`` to provide the electronic, nuclear and total energy components in
    the result object.
    """

    def __init__(
        self,
        electronic_integrals: list[ElectronicIntegrals],
        energy_shift: Optional[dict[str, complex]] = None,
        nuclear_repulsion_energy: Optional[float] = None,
        reference_energy: Optional[float] = None,
    ) -> None:
        # pylint: disable=line-too-long
        """
        Args:
            electronic_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                :class:`~qiskit_nature.second_q.properties.integrals.ElectronicIntegrals`.
            energy_shift: an optional dictionary of energy shifts.
            nuclear_repulsion_energy: the optional nuclear repulsion energy.
            reference_energy: an optional reference energy (such as the HF energy).
        """
        self._electronic_integrals: dict[ElectronicBasis, dict[int, ElectronicIntegrals]] = {}
        for integral in electronic_integrals:
            self.add_electronic_integral(integral)
        self._shift = energy_shift or {}
        self._nuclear_repulsion_energy = nuclear_repulsion_energy
        self._reference_energy = reference_energy

        # Additional, purely informational data (i.e. currently not used by the Stack itself).
        self._orbital_energies: np.ndarray = None
        self._kinetic: ElectronicIntegrals = None
        self._overlap: ElectronicIntegrals = None

    @property
    def nuclear_repulsion_energy(self) -> Optional[float]:
        """Returns the nuclear repulsion energy."""
        return self._nuclear_repulsion_energy

    @nuclear_repulsion_energy.setter
    def nuclear_repulsion_energy(self, nuclear_repulsion_energy: Optional[float]) -> None:
        """Sets the nuclear repulsion energy."""
        self._nuclear_repulsion_energy = nuclear_repulsion_energy

    @property
    def reference_energy(self) -> Optional[float]:
        """Returns the reference energy."""
        return self._reference_energy

    @reference_energy.setter
    def reference_energy(self, reference_energy: Optional[float]) -> None:
        """Sets the reference energy."""
        self._reference_energy = reference_energy

    @property
    def orbital_energies(self) -> Optional[np.ndarray]:
        """Returns the orbital energies.

        If no spin-distinction is made, this is a 1-D array, otherwise it is a 2-D array.
        """
        return self._orbital_energies

    @orbital_energies.setter
    def orbital_energies(self, orbital_energies: Optional[np.ndarray]) -> None:
        """Sets the orbital energies."""
        self._orbital_energies = orbital_energies

    @property
    def kinetic(self) -> Optional[ElectronicIntegrals]:
        """Returns the AO kinetic integrals."""
        return self._kinetic

    @kinetic.setter
    def kinetic(self, kinetic: Optional[ElectronicIntegrals]) -> None:
        """Sets the AO kinetic integrals."""
        self._kinetic = kinetic

    @property
    def overlap(self) -> Optional[ElectronicIntegrals]:
        """Returns the AO overlap integrals."""
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: Optional[ElectronicIntegrals]) -> None:
        """Sets the AO overlap integrals."""
        self._overlap = overlap

    # pylint: disable=invalid-name
    @classmethod
    def from_raw_integrals(
        cls,
        basis: ElectronicBasis,
        h1: np.ndarray,
        h2: np.ndarray,
        h1_b: Optional[np.ndarray] = None,
        h2_bb: Optional[np.ndarray] = None,
        h2_ba: Optional[np.ndarray] = None,
        threshold: float = ElectronicIntegrals.INTEGRAL_TRUNCATION_LEVEL,
    ) -> ElectronicEnergy:
        """Construct an ``ElectronicEnergy`` from raw integrals in a given basis.

        When setting the basis to
        :class:`~qiskit_nature.second_q.properties.bases.ElectronicBasis.SO`,
        all of the arguments ``h1_b``, ``h2_bb`` and ``h2_ba`` will be ignored.

        Args:
            basis: the
                :class:`~qiskit_nature.second_q.properties.bases.ElectronicBasis`
                of the provided integrals.
            h1: the one-body integral matrix.
            h2: the two-body integral matrix.
            h1_b: the optional beta-spin one-body integral matrix.
            h2_bb: the optional beta-beta-spin two-body integral matrix.
            h2_ba: the optional beta-alpha-spin two-body integral matrix.
            threshold: the truncation level below which to treat the integral in the SO matrix as
                zero-valued.

        Returns:
            An instance of this property.
        """
        if basis == ElectronicBasis.SO:
            one_body = OneBodyElectronicIntegrals(basis, h1, threshold=threshold)
            two_body = TwoBodyElectronicIntegrals(basis, h2, threshold=threshold)
        else:
            one_body = OneBodyElectronicIntegrals(basis, (h1, h1_b), threshold=threshold)
            h2_ab: Optional[np.ndarray] = h2_ba.T if h2_ba is not None else None
            two_body = TwoBodyElectronicIntegrals(
                basis, (h2, h2_ba, h2_bb, h2_ab), threshold=threshold
            )

        return cls([one_body, two_body])

    def add_electronic_integral(self, integral: ElectronicIntegrals) -> None:
        """Adds an ElectronicIntegrals instance to the internal storage.

        Internally, the ElectronicIntegrals are stored in a nested dictionary sorted by their basis
        and number of body terms. This simplifies access based on these properties (see
        ``get_electronic_integral``) and avoids duplicate, inconsistent entries.

        Args:
            integral: the ElectronicIntegrals to add.
        """
        if integral._basis not in self._electronic_integrals:
            self._electronic_integrals[integral._basis] = {}
        self._electronic_integrals[integral._basis][integral._num_body_terms] = integral

    def get_electronic_integral(
        self, basis: ElectronicBasis, num_body_terms: int
    ) -> Optional[ElectronicIntegrals]:
        """Gets an ElectronicIntegrals given the basis and number of body terms.

        Args:
            basis: the ElectronicBasis of the queried integrals.
            num_body_terms: the number of body terms of the queried integrals.

        Returns:
            The queried integrals object (or None if unavailable).
        """
        ints_basis = self._electronic_integrals.get(basis, None)
        if ints_basis is None:
            return None
        return ints_basis.get(num_body_terms, None)

    def transform_basis(self, transform: ElectronicBasisTransform) -> None:
        """Applies an ElectronicBasisTransform to the internal integrals.

        Args:
            transform: the ElectronicBasisTransform to apply.
        """
        for integral in self._electronic_integrals[transform.initial_basis].values():
            self.add_electronic_integral(integral.transform_basis(transform))

    def integral_operator(self, density: OneBodyElectronicIntegrals) -> OneBodyElectronicIntegrals:
        """Constructs the Fock operator resulting from this
        :class:`~qiskit_nature.second_q.hamiltonians.ElectronicEnergy`.

        Args:
            density: the electronic density at which to compute the operator.

        Returns:
            OneBodyElectronicIntegrals: the operator stored as ElectronicIntegrals.

        Raises:
            NotImplementedError: if no AO electronic integrals are available.
        """
        if ElectronicBasis.AO not in self._electronic_integrals:
            raise NotImplementedError(
                "Construction of the Fock operator outside of the AO basis is not yet implemented."
            )

        one_e_ints = self.get_electronic_integral(ElectronicBasis.AO, 1)
        two_e_ints = cast(
            TwoBodyElectronicIntegrals, self.get_electronic_integral(ElectronicBasis.AO, 2)
        )

        op = one_e_ints

        coulomb = two_e_ints.compose(density, "ijkl,ji->kl")
        coulomb_inv = OneBodyElectronicIntegrals(
            ElectronicBasis.AO, (coulomb.get_matrix(1), coulomb.get_matrix(0))
        )
        exchange = two_e_ints.compose(density, "ijkl,jk->il")
        op += coulomb + coulomb_inv - exchange

        return cast(OneBodyElectronicIntegrals, op)

    @property
    def register_length(self) -> int:
        ints = None
        if ElectronicBasis.SO in self._electronic_integrals:
            ints = self._electronic_integrals[ElectronicBasis.SO]
        elif ElectronicBasis.MO in self._electronic_integrals:
            ints = self._electronic_integrals[ElectronicBasis.MO]

        return len(list(ints.values())[0].to_spin())

    def second_q_op(self) -> FermionicOp:
        """Returns the second quantized operator constructed from the contained electronic integrals.

        Returns:
            A `dict` of `FermionicOp` objects.
        """
        ints = None
        if ElectronicBasis.SO in self._electronic_integrals:
            ints = self._electronic_integrals[ElectronicBasis.SO]
        elif ElectronicBasis.MO in self._electronic_integrals:
            ints = self._electronic_integrals[ElectronicBasis.MO]

        op = cast(FermionicOp, sum(int.to_second_q_op() for int in ints.values()))

        return op

    def interpret(self, result: "EigenstateResult") -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.hartree_fock_energy = self._reference_energy
        result.nuclear_repulsion_energy = self._nuclear_repulsion_energy
        result.extracted_transformer_energies = self._shift.copy()
