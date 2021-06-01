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

from __future__ import annotations

from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from .electronic_integrals import _1BodyElectronicIntegrals
from .property import Property


class Magnetization(Property):
    """TODO."""

    def __init__(
        self,
        register_length: int,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__, register_length)

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> Magnetization:
        """TODO."""
        if isinstance(result, WatsonHamiltonian):
            raise QiskitNatureError("TODO.")

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """TODO."""
        matrix = np.eye(self.register_length // 2, dtype=complex) * 0.5
        matrix[self.register_length // 4 :, self.register_length // 4 :] *= -1.0
        ints = _1BodyElectronicIntegrals((matrix, None))
        return [ints.to_second_q_op()]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass
