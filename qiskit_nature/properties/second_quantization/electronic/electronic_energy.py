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

from typing import Dict, List, Optional, cast

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult, LegacyElectronicDriverResult
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
        nuclear_repulsion_energy: Optional[complex] = None,
        reference_energy: Optional[complex] = None,
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

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "ElectronicEnergy":
        """Construct an ElectronicEnergy instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, LegacyElectronicDriverResult)

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

        return cls(
            integrals,
            energy_shift=energy_shift,
            nuclear_repulsion_energy=qmol.nuclear_repulsion_energy,
            reference_energy=qmol.hf_energy,
        )

    def matrix_operator(self, density: OneBodyElectronicIntegrals) -> OneBodyElectronicIntegrals:
        """Constructs the operator of this property in matrix-format for a given density.

        An IntegralProperty typically represents an observable which can be expressed in terms of a
        matrix-formatted operator at a given electronic density. This method must be implemented by
        a subclass to provide this functionality.

        Args:
            density: the electronic density at which to compute the matrix operator.

        Returns:
            OneBodyElectronicIntegrals: the matrix-formatted operator stored as ElectronicIntegrals.

        Raises:
            NotImplementedError: if no AO-integrals are stored within `self`.
        """
        if ElectronicBasis.AO not in self._electronic_integrals.keys():
            raise NotImplementedError()

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
