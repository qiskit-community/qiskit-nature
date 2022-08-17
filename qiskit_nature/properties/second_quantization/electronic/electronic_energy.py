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

from typing import Optional, cast

import h5py
import numpy as np

from qiskit_nature.drivers import QMolecule
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .bases import ElectronicBasis
from .integrals import (
    ElectronicIntegrals,
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from ....deprecation import (
    deprecate_method,
    warn_deprecated,
    DeprecatedType,
    NatureDeprecationWarning,
)


class ElectronicEnergy(IntegralProperty):
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
                :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals`.
            energy_shift: an optional dictionary of energy shifts.
            nuclear_repulsion_energy: the optional nuclear repulsion energy.
            reference_energy: an optional reference energy (such as the HF energy).
        """
        super().__init__(self.__class__.__name__, electronic_integrals, shift=energy_shift)
        self._nuclear_repulsion_energy = nuclear_repulsion_energy
        self._reference_energy = reference_energy

        # Additional, purely informational data (i.e. currently not used by the Stack itself).
        self._orbital_energies: np.ndarray = None
        self._kinetic: ElectronicIntegrals = None
        self._overlap: ElectronicIntegrals = None
        warn_deprecated(
            "0.5.0",
            old_type=DeprecatedType.CLASS,
            old_name="qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy",
            new_type=DeprecatedType.CLASS,
            new_name="qiskit_nature.second_q.hamiltonians.ElectronicEnergy",
            category=NatureDeprecationWarning,
        )

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        if self.nuclear_repulsion_energy is not None:
            group.attrs["nuclear_repulsion_energy"] = self.nuclear_repulsion_energy

        if self.reference_energy is not None:
            group.attrs["reference_energy"] = self.reference_energy

        if self.orbital_energies is not None:
            group.attrs["orbital_energies"] = self.orbital_energies

        if self.kinetic is not None:
            kinetic_group = group.create_group("kinetic")
            self.kinetic.to_hdf5(kinetic_group)

        if self.overlap is not None:
            overlap_group = group.create_group("overlap")
            self.overlap.to_hdf5(overlap_group)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> ElectronicEnergy:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        integral_property = IntegralProperty.from_hdf5(h5py_group)

        ret = ElectronicEnergy(list(integral_property), energy_shift=integral_property._shift)

        ret.nuclear_repulsion_energy = h5py_group.attrs.get("nuclear_repulsion_energy", None)
        ret.reference_energy = h5py_group.attrs.get("reference_energy", None)
        ret.orbital_energies = h5py_group.attrs.get("orbital_energies", None)

        if "kinetic" in h5py_group.keys():
            ret.kinetic = ElectronicIntegrals.from_hdf5(
                h5py_group["kinetic"]["OneBodyElectronicIntegrals"]
            )

        if "overlap" in h5py_group.keys():
            ret.overlap = ElectronicIntegrals.from_hdf5(
                h5py_group["overlap"]["OneBodyElectronicIntegrals"]
            )

        return ret

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

    @classmethod
    @deprecate_method("0.4.0")
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> ElectronicEnergy:
        """Construct an ``ElectronicEnergy`` instance from a :class:`~qiskit_nature.drivers.QMolecule`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.drivers.QMolecule` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.WatsonHamiltonian` is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        energy_shift = qmol.energy_shift.copy()

        integrals: list[ElectronicIntegrals] = []
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
        :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`,
        all of the arguments ``h1_b``, ``h2_bb`` and ``h2_ba`` will be ignored.

        Args:
            basis: the
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis`
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

    def integral_operator(self, density: OneBodyElectronicIntegrals) -> OneBodyElectronicIntegrals:
        """Constructs the Fock operator resulting from this
        :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy`.

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

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.hartree_fock_energy = self._reference_energy
        result.nuclear_repulsion_energy = self._nuclear_repulsion_energy
        result.extracted_transformer_energies = self._shift.copy()
