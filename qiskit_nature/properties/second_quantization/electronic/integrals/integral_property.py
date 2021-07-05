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

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from qiskit_nature.operators.second_quantization import FermionicOp

from ...second_quantized_property import LegacyDriverResult, SecondQuantizedProperty
from ..bases import ElectronicBasis, ElectronicBasisTransform
from .electronic_integrals import ElectronicIntegrals
from .one_body_electronic_integrals import OneBodyElectronicIntegrals


class IntegralProperty(SecondQuantizedProperty):
    """A common Property object based on `ElectronicIntegrals` as its raw data.

    This is a common base class, extracted to be used by (at the time of writing) the
    `ElectronicEnergy` and the `DipoleMoment` properties. More subclasses may be added in the
    future.
    """

    def __init__(
        self,
        name: str,
        electronic_integrals: List[ElectronicIntegrals],
        shift: Optional[Dict[str, complex]] = None,
    ):
        """
        Args:
            basis: the basis which the integrals in ``electronic_integrals`` are stored in.
            electronic_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                ``ElectronicIntegrals``.
            shift: an optional dictionary of value shifts.
        """
        super().__init__(name)
        self._electronic_integrals: Dict[ElectronicBasis, Dict[int, ElectronicIntegrals]] = {}
        for integral in electronic_integrals:
            self.add_electronic_integral(integral)
        self._shift = shift or {}

    def __repr__(self) -> str:
        string = super().__repr__()
        for basis_ints in self._electronic_integrals.values():
            for ints in basis_ints.values():
                string += f"\n\t{ints}"
        if self._shift:
            string += "\n\tEnergy Shifts:"
            for name, shift in self._shift.items():
                string += f"\n\t\t{name} = {shift}"
        return string

    def add_electronic_integral(self, integral: ElectronicIntegrals) -> None:
        """Adds an ElectronicIntegrals instance to the internal storage.

        Internally, the ElectronicIntegrals are stored in a nested dictionary sorted by their basis
        and number of body terms. This simplifies access based on these properties (see
        `get_electronic_integral`) and avoids duplicate, inconsistent entries.

        Args:
            integral: the ElectronicIntegrals to add.
        """
        if integral._basis not in self._electronic_integrals.keys():
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

    @abstractmethod
    def matrix_operator(self, density: OneBodyElectronicIntegrals) -> OneBodyElectronicIntegrals:
        """TODO."""

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the Hamiltonian constructed by the stored electronic integrals."""
        ints = None
        if ElectronicBasis.SO in self._electronic_integrals.keys():
            ints = self._electronic_integrals[ElectronicBasis.SO]
        elif ElectronicBasis.MO in self._electronic_integrals.keys():
            ints = self._electronic_integrals[ElectronicBasis.MO]
        return [sum(int.to_second_q_op() for int in ints.values()).reduce()]  # type: ignore

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "IntegralProperty":
        """This property does not support construction from a driver result (yet).

        Args:
            result: ignored.

        Raises:
            NotImplemented
        """
        raise NotImplementedError()
