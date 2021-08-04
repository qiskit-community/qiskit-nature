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

"""The ElectronicEnergy property."""

from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from qiskit_nature.drivers import QMolecule
from qiskit_nature.results import EigenstateResult

from .bases import ElectronicBasis
from .integrals import (
    ElectronicIntegrals,
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)


class ElectronicEnergy(IntegralProperty):
    """The ElectronicEnergy property.

    This is the main property of any electronic structure problem. It constructs the Hamiltonian
    whose eigenvalue is the target of a later used Quantum algorithm.

    Note that this Property computes **purely** the electronic energy (possibly minus additional
    shifts introduced via e.g. classical transformers). However, for convenience it provides a
    storage location for the nuclear repulsion energy. If available, this information will be used
    during the call of `interpret` to provide the electronic, nuclear and total energy components in
    the result object.
    """

    def __init__(
        self,
        electronic_integrals: List[ElectronicIntegrals],
        energy_shift: Optional[Dict[str, complex]] = None,
        nuclear_repulsion_energy: Optional[float] = None,
        reference_energy: Optional[float] = None,
    ):
        """
        Args:
            basis: the basis which the integrals in ``electronic_integrals`` are stored in.
            electronic_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                ``ElectronicIntegrals``.
            reference_energy: an optional reference energy (such as the HF energy).
            energy_shift: an optional dictionary of energy shifts.
        """
        super().__init__(self.__class__.__name__, electronic_integrals, shift=energy_shift)
        self._nuclear_repulsion_energy = nuclear_repulsion_energy
        self._reference_energy = reference_energy

        # Additional, purely information data (i.e. currently not used by the Stack itself).
        self._orbital_enerfies: np.ndarray = None
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
    def orbital_energies(self) -> np.ndarray:
        """Returns the orbital energies.

        If no spin-distinction is made, this is a 1-D array, otherwise it is a 2-D array.
        """
        return self._orbital_energies

    @orbital_energies.setter
    def orbital_energies(self, orbital_energies: np.ndarray) -> None:
        """Sets the orbital energies."""
        self._orbital_energies = orbital_energies

    @property
    def kinetic(self) -> ElectronicIntegrals:
        """Returns the AO kinetic integrals."""
        return self._kinetic

    @kinetic.setter
    def kinetic(self, kinetic: ElectronicIntegrals) -> None:
        """Sets the AO kinetic integrals."""
        self._kinetic = kinetic

    @property
    def overlap(self) -> ElectronicIntegrals:
        """Returns the AO overlap integrals."""
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: ElectronicIntegrals) -> None:
        """Sets the AO overlap integrals."""
        self._overlap = overlap

    @classmethod
    def from_legacy_driver_result(cls, result: Any) -> "ElectronicEnergy":
        """Construct an ElectronicEnergy instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        energy_shift = qmol.energy_shift.copy()

        integrals: List[ElectronicIntegrals] = []
        if qmol.hcore is not None:
            integrals.append(
                OneBodyElectronicIntegrals(ElectronicBasis.AO, (qmol.hcore, qmol.hcore_b))
            )
        if qmol.eri is not None:
            integrals.append(
                TwoBodyElectronicIntegrals(ElectronicBasis.AO, (qmol.eri, None, None, None))
            )
        if qmol.mo_onee_ints is not None:
            integrals.append(
                OneBodyElectronicIntegrals(
                    ElectronicBasis.MO, (qmol.mo_onee_ints, qmol.mo_onee_ints_b)
                )
            )
        if qmol.mo_eri_ints is not None:
            integrals.append(
                TwoBodyElectronicIntegrals(
                    ElectronicBasis.MO,
                    (qmol.mo_eri_ints, qmol.mo_eri_ints_ba, qmol.mo_eri_ints_bb, None),
                )
            )

        ret = cls(
            integrals,
            energy_shift=energy_shift,
            nuclear_repulsion_energy=qmol.nuclear_repulsion_energy,
            reference_energy=qmol.hf_energy,
        )

        orb_energies = qmol.orbital_energies
        if qmol.orbital_energies_b is not None:
            orb_energies = np.asarray((qmol.orbital_energies, qmol.orbital_energies_b))
        ret.orbital_energies = orb_energies

        if qmol.kinetic is not None:
            ret.kinetic = OneBodyElectronicIntegrals(ElectronicBasis.AO, (qmol.kinetic, None))

        if qmol.overlap is not None:
            ret.overlap = OneBodyElectronicIntegrals(ElectronicBasis.AO, (qmol.overlap, None))

        return ret

    def integral_operator(self, density: OneBodyElectronicIntegrals) -> OneBodyElectronicIntegrals:
        """Constructs the Fock operator resulting from this `ElectronicEnergy`.

        Args:
            density: the electronic density at which to compute the operator.

        Returns:
            OneBodyElectronicIntegrals: the operator stored as ElectronicIntegrals.

        Raises:
            NotImplementedError: if no AO electronic integrals are available.
        """
        if ElectronicBasis.AO not in self._electronic_integrals.keys():
            raise NotImplementedError(
                "Construction of the Fock operator outside of the AO basis is not yet implemented."
            )

        one_e_ints = self.get_electronic_integral(ElectronicBasis.AO, 1)
        two_e_ints = cast(
            TwoBodyElectronicIntegrals, self.get_electronic_integral(ElectronicBasis.AO, 2)
        )

        op = one_e_ints

        coulomb = two_e_ints.compose(density, "ijkl,ji->kl")
        # by reversing the order of the matrices we can construct the (beta, alpha)-ordered Coulomb
        # integrals
        coulomb_inv = OneBodyElectronicIntegrals(ElectronicBasis.AO, coulomb._matrices[::-1])
        exchange = two_e_ints.compose(density, "ijkl,jk->il")
        op += coulomb + coulomb_inv - exchange

        return cast(OneBodyElectronicIntegrals, op)

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.hartree_fock_energy = self._reference_energy
        result.nuclear_repulsion_energy = self._nuclear_repulsion_energy
        result.extracted_transformer_energies = self._shift.copy()
