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

"""TODO."""

from typing import cast, Dict, List, Optional, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from .bases import VibrationalBasis
from .integrals import VibrationalIntegrals
from ..property import Property


class VibrationalEnergy(Property):
    """TODO."""

    def __init__(
        self,
        vibrational_integrals: Dict[int, VibrationalIntegrals],
        truncation_order: Optional[int] = None,
        basis: Optional[VibrationalBasis] = None,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._vibrational_integrals = vibrational_integrals
        self._truncation_order = truncation_order
        self._basis = basis

    @property
    def truncation_order(self) -> int:
        """Returns the truncation_order."""
        return self._truncation_order

    @truncation_order.setter
    def truncation_order(self, truncation_order: int) -> None:
        """Sets the truncation_order."""
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
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "VibrationalEnergy":
        """TODO."""
        if isinstance(result, QMolecule):
            raise QiskitNatureError("TODO.")

        w_h = cast(WatsonHamiltonian, result)

        vib_ints: Dict[int, VibrationalIntegrals] = {
            1: VibrationalIntegrals(1, []),
            2: VibrationalIntegrals(2, []),
            3: VibrationalIntegrals(3, []),
        }
        for coeff, *indices in w_h.data:
            ints = [int(i) for i in indices]
            num_body = len(set(ints))
            vib_ints[num_body]._integrals.append((coeff, tuple(ints)))

        return cls(vib_ints)

    def second_q_ops(self) -> List[VibrationalOp]:
        """TODO."""
        ops = []
        for num_body, ints in self._vibrational_integrals.items():
            if self._truncation_order is not None and num_body > self._truncation_order:
                break
            ints.basis = self.basis
            ops.append(ints.to_second_q_op())
        return [sum(ops)]  # type: ignore

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass
