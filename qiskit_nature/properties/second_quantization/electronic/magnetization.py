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

from typing import cast, List, Union

from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .types import ElectronicProperty


class Magnetization(ElectronicProperty):
    """The Magnetization property."""

    def __init__(self, num_spin_orbitals: int) -> None:
        """
        Args:
            num_spin_orbitals: the number of spin orbitals in the system.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_spin_orbitals} SOs"]
        return "\n".join(string)

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "Magnetization":
        """Construct a Magnetization instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

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

    # TODO: refactor after closing https://github.com/Qiskit/qiskit-terra/issues/6772
    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.magnetization = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            if aux_op_eigenvalues[2] is not None:
                result.magnetization.append(aux_op_eigenvalues[2][0].real)  # type: ignore
            else:
                result.magnetization.append(None)
