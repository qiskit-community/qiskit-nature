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

from typing import cast, Dict, List, Optional

from qiskit_nature.drivers.second_quantization import WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from .bases import VibrationalBasis
from .integrals import VibrationalIntegrals
from ..property import DriverResult, Property, VibrationalDriverResult


class VibrationalEnergy(Property):
    """The VibrationalEnergy property.

    This is the main property of any vibrational structure problem. It constructs the Hamiltonian
    whose eigenvalue is the target of a later used Quantum algorithm.
    """

    def __init__(
        self,
        vibrational_integrals: Dict[int, VibrationalIntegrals],
        truncation_order: Optional[int] = None,
        basis: Optional[VibrationalBasis] = None,
    ):
        """
        Args:
            vibrational_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                ``VibrationalIntegrals``.
            truncation_order: an optional truncation order for the highest number of body terms to
                include in the constructed Hamiltonian.
            basis: the ``VibrationalBasis`` through which to map the integrals into second
                quantization. This property **MUST** be set before the second-quantized operator can
                be constructed.
        """
        super().__init__(self.__class__.__name__)
        self._vibrational_integrals = vibrational_integrals
        self._truncation_order = truncation_order
        self._basis = basis

    @property
    def truncation_order(self) -> int:
        """Returns the truncation order."""
        return self._truncation_order

    @truncation_order.setter
    def truncation_order(self, truncation_order: int) -> None:
        """Sets the truncation order."""
        self._truncation_order = truncation_order

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @classmethod
    def from_driver_result(cls, result: DriverResult) -> "VibrationalEnergy":
        """Construct a VibrationalEnergy instance from a WatsonHamiltonian.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                WatsonHamiltonian is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a QMolecule is provided.
        """
        cls._validate_input_type(result, VibrationalDriverResult)

        w_h = cast(WatsonHamiltonian, result)

        vib_ints: Dict[int, VibrationalIntegrals] = {
            1: VibrationalIntegrals(1, []),
            2: VibrationalIntegrals(2, []),
            3: VibrationalIntegrals(3, []),
        }
        for coeff, *indices in w_h.data:
            ints = [int(i) for i in indices]
            num_body = len(set(ints))
            vib_ints[num_body].integrals.append((coeff, tuple(ints)))

        return cls(vib_ints)

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
        """Interprets an `qiskit_nature.result.EigenstateResult` in the context of this Property.

        This is currently a method stub which may be used in the future.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
