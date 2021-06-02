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

from typing import cast, List, Union

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from .electronic_integrals import Basis, _1BodyElectronicIntegrals
from .property import Property


class Magnetization(Property):
    """TODO."""

    def __init__(
        self,
        num_spin_orbitals: int,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "Magnetization":
        """TODO."""
        if isinstance(result, WatsonHamiltonian):
            raise QiskitNatureError("TODO.")

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """TODO."""
        matrix_a = np.eye(self._num_spin_orbitals // 2, dtype=complex) * 0.5
        matrix_b = -1.0 * matrix_a.copy()
        ints = _1BodyElectronicIntegrals(Basis.MO, (matrix_a, matrix_b))
        return [ints.to_second_q_op()]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass
