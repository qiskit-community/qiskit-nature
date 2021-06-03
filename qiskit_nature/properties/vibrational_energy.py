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

from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.results import EigenstateResult

from .vibrational_integrals import (
    _VibrationalIntegrals,
    _1BodyVibrationalIntegrals,
    _2BodyVibrationalIntegrals,
    _3BodyVibrationalIntegrals,
    BosonicBasis,
)
from .property import Property


class VibrationalEnergy(Property):
    """TODO."""

    def __init__(
        self,
        vibrational_integrals: Dict[int, _VibrationalIntegrals],
        truncation_order: Optional[int] = None,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._vibrational_integrals = vibrational_integrals
        self._truncation_order = truncation_order
        self._basis: BosonicBasis = None

    @property
    def truncation_order(self) -> int:
        """Returns the truncation_order."""
        return self._truncation_order

    @truncation_order.setter
    def truncation_order(self, truncation_order: int) -> None:
        """Sets the truncation_order."""
        self._truncation_order = truncation_order

    @property
    def basis(self) -> BosonicBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: BosonicBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "VibrationalEnergy":
        """TODO."""
        if isinstance(result, QMolecule):
            raise QiskitNatureError("TODO.")

        w_h = cast(WatsonHamiltonian, result)

        # TODO: construct empty _VibrationalIntegrals and append to their lists
        parsed: Dict[int, List[Tuple[float, Tuple[int, ...]]]] = {}
        for coeff, *indices in w_h.data:
            ints = [int(i) for i in indices]
            num_body = len(set(ints))
            if num_body not in parsed.keys():
                parsed[num_body] = []
            parsed[num_body].append((coeff, tuple(ints)))

        vib_ints: Dict[int, _VibrationalIntegrals] = {}
        if 1 in parsed.keys():
            vib_ints[1] = _1BodyVibrationalIntegrals(1, parsed[1])
        if 2 in parsed.keys():
            vib_ints[2] = _2BodyVibrationalIntegrals(2, parsed[2])
        if 3 in parsed.keys():
            vib_ints[3] = _3BodyVibrationalIntegrals(3, parsed[3])

        return cls(vib_ints)

    def second_q_ops(self) -> List[VibrationalOp]:
        """TODO."""
        # TODO: limit to truncation_order
        self._vibrational_integrals[1].basis = self.basis
        op = self._vibrational_integrals[1].to_second_q_op()
        self._vibrational_integrals[2].basis = self.basis
        op += self._vibrational_integrals[2].to_second_q_op()
        self._vibrational_integrals[3].basis = self.basis
        op += self._vibrational_integrals[3].to_second_q_op()
        return [op]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass
