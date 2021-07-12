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

"""The VibrationalEnergy property."""

from typing import cast, Dict, List, Optional, Tuple

from qiskit_nature.drivers.second_quantization import WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from .bases import VibrationalBasis
from .integrals import VibrationalIntegrals
from .vibrational_property import VibrationalProperty
from ..second_quantized_property import LegacyDriverResult, LegacyVibrationalDriverResult


class VibrationalEnergy(VibrationalProperty):
    """The VibrationalEnergy property.

    This is the main property of any vibrational structure problem. It constructs the Hamiltonian
    whose eigenvalue is the target of a later used Quantum algorithm.
    """

    def __init__(
        self,
        vibrational_integrals: List[VibrationalIntegrals],
        truncation_order: Optional[int] = None,
        basis: Optional[VibrationalBasis] = None,
    ):
        """
        Args:
            vibrational_integrals: a list of ``VibrationalIntegrals``.
            truncation_order: an optional truncation order for the highest number of body terms to
                include in the constructed Hamiltonian.
            basis: the ``VibrationalBasis`` through which to map the integrals into second
                quantization. This property **MUST** be set before the second-quantized operator can
                be constructed.
        """
        super().__init__(self.__class__.__name__, basis)
        self._vibrational_integrals: Dict[int, VibrationalIntegrals] = {}
        for integral in vibrational_integrals:
            self.add_vibrational_integral(integral)
        self._truncation_order = truncation_order

    @property
    def truncation_order(self) -> int:
        """Returns the truncation order."""
        return self._truncation_order

    @truncation_order.setter
    def truncation_order(self, truncation_order: int) -> None:
        """Sets the truncation order."""
        self._truncation_order = truncation_order

    def __repr__(self) -> str:
        string = [super().__repr__()]
        for ints in self._vibrational_integrals.values():
            string += [f"\t{ints}"]
        return "\n".join(string)

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "VibrationalEnergy":
        """Construct a VibrationalEnergy instance from a WatsonHamiltonian.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                WatsonHamiltonian is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a QMolecule is provided.
        """
        cls._validate_input_type(result, LegacyVibrationalDriverResult)

        w_h = cast(WatsonHamiltonian, result)

        sorted_integrals: Dict[int, List[Tuple[float, Tuple[int, ...]]]] = {1: [], 2: [], 3: []}
        for coeff, *indices in w_h.data:
            ints = [int(i) for i in indices]
            num_body = len(set(ints))
            sorted_integrals[num_body].append((coeff, tuple(ints)))

        return cls(
            [VibrationalIntegrals(num_body, ints) for num_body, ints in sorted_integrals.items()]
        )

    def add_vibrational_integral(self, integral: VibrationalIntegrals) -> None:
        """Adds a VibrationalIntegrals instance to the internal storage.

        Internally, the VibrationalIntegrals are stored in a dictionary sorted by their number of
        body terms. This simplifies access based on these properties (see
        `get_vibrational_integral`) and avoids duplicate, inconsistent entries.

        Args:
            integral: the VibrationalIntegrals to add.
        """
        self._vibrational_integrals[integral._num_body_terms] = integral

    def get_vibrational_integral(self, num_body_terms: int) -> Optional[VibrationalIntegrals]:
        """Gets an VibrationalIntegrals given the number of body terms.

        Args:
            num_body_terms: the number of body terms of the queried integrals.

        Returns:
            The queried integrals object (or None if unavailable).
        """
        return self._vibrational_integrals.get(num_body_terms, None)

    def second_q_ops(self) -> List[VibrationalOp]:
        """Returns a list containing the Hamiltonian constructed by the stored integrals."""
        ops = []
        for num_body, ints in self._vibrational_integrals.items():
            if self._truncation_order is not None and num_body > self._truncation_order:
                break
            ints.basis = self.basis
            ops.append(ints.to_second_q_op())
        return [sum(ops)]  # type: ignore

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
