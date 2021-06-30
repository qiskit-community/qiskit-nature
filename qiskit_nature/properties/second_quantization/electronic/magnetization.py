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

"""The Magnetization property."""

from typing import cast, List

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp

from ..second_quantized_property import (
    DriverResult,
    ElectronicDriverResult,
    SecondQuantizedProperty,
)


class Magnetization(SecondQuantizedProperty):
    """The Magnetization property."""

    def __init__(self, num_spin_orbitals: int):
        """
        Args:
            num_spin_orbitals: the number of spin orbitals in the system.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals

    @classmethod
    def from_driver_result(cls, result: DriverResult) -> "Magnetization":
        """Construct a Magnetization instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, ElectronicDriverResult)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

    def reduce_system_size(self, active_orbital_indices: List[int]) -> "Magnetization":
        """TODO."""
        return Magnetization(len(active_orbital_indices) * 2)

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the magnetization operator."""
        op = FermionicOp(
            [
                (f"N_{o}", 0.5 if o < self._num_spin_orbitals // 2 else -0.5)
                for o in range(self._num_spin_orbitals)
            ],
            register_length=self._num_spin_orbitals,
        )
        return [op]
