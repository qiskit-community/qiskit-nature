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

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..property import Property


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
        op = FermionicOp(
            [
                (f"N_{o}", 0.5 if o < self._num_spin_orbitals // 2 else -0.5)
                for o in range(self._num_spin_orbitals)
            ],
            register_length=self._num_spin_orbitals,
        )
        return [op]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass
