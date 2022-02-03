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

"""The IntegralProperty property."""

from __future__ import annotations

from typing import Dict, Generator, List, Optional

import h5py

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ...second_quantized_property import LegacyDriverResult
from ..bases import ElectronicBasis, ElectronicBasisTransform
from ..types import ElectronicProperty
from .electronic_integrals import ElectronicIntegrals
from .one_body_electronic_integrals import OneBodyElectronicIntegrals


class IntegralProperty(ElectronicProperty):
    """A common Property object based on
    :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals`
    as its raw data.

    This is a common base class, extracted to be used by (at the time of writing) the
    :class:`~qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy` and the
    :class:`~qiskit_nature.properties.second_quantization.electronic.DipoleMoment` properties. More
    subclasses may be added in the future.
    """

    def __init__(
        self,
        name: str,
        electronic_integrals: List[ElectronicIntegrals],
        shift: Optional[Dict[str, complex]] = None,
    ) -> None:
        # pylint: disable=line-too-long
        """
        Args:
            name: the name of this Property object.
            electronic_integrals: a list of
                :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals`.
            shift: an optional dictionary of value shifts.
        """
        super().__init__(name)
        self._electronic_integrals: Dict[ElectronicBasis, Dict[int, ElectronicIntegrals]] = {}
        for integral in electronic_integrals:
            self.add_electronic_integral(integral)
        self._shift = shift or {}

    def __str__(self) -> str:
        string = [super().__str__()]
        for ints in self:
            string += ["\t" + "\n\t".join(str(ints).split("\n"))]
        if self._shift:
            string += ["\tEnergy Shifts:"]
            for name, shift in self._shift.items():
                string += [f"\t\t{name} = {shift}"]
        return "\n".join(string)

    def __iter__(self) -> Generator[ElectronicIntegrals, None, None]:
        """Returns the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[ElectronicIntegrals, None, None]:
        """A generator-iterator method [1] iterating over all internal ``ElectronicIntegrals``.

        [1]: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
        """
        for basis_ints in self._electronic_integrals.values():
            for ints in basis_ints.values():
                yield ints

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
        """Constructs the operator of this property in integral-format for a given density.

        An IntegralProperty typically represents an observable which can be expressed in terms of a
        matrix-formatted operator at a given electronic density. In the Property framework the
        generic representation of such matrices are
        :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals`.

        Args:
            density: the electronic density at which to compute the operator.

        Returns:
            OneBodyElectronicIntegrals: the operator stored as ElectronicIntegrals.

        Raises:
            NotImplementedError: this method is not implemented by the base class. It cannot be made
                                 abstract because we need other functionality of the base class
                                 available on its own.
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__}.integral_operator is not implemented!"
        )

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns the second quantized operator constructed from the contained electronic integrals.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `FermionicOp` objects.
        """
        ints = None
        if ElectronicBasis.SO in self._electronic_integrals:
            ints = self._electronic_integrals[ElectronicBasis.SO]
        elif ElectronicBasis.MO in self._electronic_integrals:
            ints = self._electronic_integrals[ElectronicBasis.MO]

        op = sum(int.to_second_q_op() for int in ints.values()).reduce()  # type: ignore[union-attr]

        if not settings.dict_aux_operators:
            return [op]

        return {self.name: op}

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> IntegralProperty:
        """This property does not support construction from a legacy driver result (yet).

        Args:
            result: ignored.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        ints_group = group.create_group("electronic_integrals")
        for basis, integrals in self._electronic_integrals.items():
            basis_group = ints_group.create_group(basis.name)
            for integral in integrals.values():
                integral.to_hdf5(basis_group)

        shift_group = group.create_group("shift")
        for name, shift in self._shift.items():
            shift_group.attrs[name] = shift

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> IntegralProperty:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        ints = []
        for basis_group in h5py_group["electronic_integrals"].values():
            for int_group in basis_group.values():
                ints.append(ElectronicIntegrals.from_hdf5(int_group))

        shifts = {}
        for name, shift in h5py_group["shift"].attrs.items():
            shifts[name] = shift

        class_name = h5py_group.attrs.get("__class__", "")

        return IntegralProperty(class_name, ints, shifts)
