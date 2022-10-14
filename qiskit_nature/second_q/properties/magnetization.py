# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Magnetization property."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit_nature.second_q.operators import FermionicOp

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import EigenstateResult


class Magnetization:
    """The Magnetization property."""

    def __init__(self, num_spin_orbitals: int) -> None:
        """
        Args:

            num_spin_orbitals: the number of spin orbitals in the system.
        """
        self._num_spin_orbitals = num_spin_orbitals

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the number of spin orbitals."""
        return self._num_spin_orbitals

    @num_spin_orbitals.setter
    def num_spin_orbitals(self, num_spin_orbitals: int) -> None:
        """Sets the number of spin orbitals."""
        self._num_spin_orbitals = num_spin_orbitals

    def second_q_ops(self) -> dict[str, FermionicOp]:
        """Returns the second quantized magnetization operator.

        Returns:
            A `dict` of `SecondQuantizedOp` objects.
        """
        op = FermionicOp(
            {
                f"+_{o} -_{o}": 0.5 if o < self._num_spin_orbitals // 2 else -0.5
                for o in range(self._num_spin_orbitals)
            },
            num_spin_orbitals=self._num_spin_orbitals,
        )

        return {self.__class__.__name__: op}

    def interpret(self, result: "EigenstateResult") -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.magnetization = []

        if not isinstance(result.aux_operators_evaluated, list):
            aux_operators_evaluated = [result.aux_operators_evaluated]
        else:
            aux_operators_evaluated = result.aux_operators_evaluated
        for aux_op_eigenvalues in aux_operators_evaluated:
            if aux_op_eigenvalues is None:
                continue

            _key = self.__class__.__name__ if isinstance(aux_op_eigenvalues, dict) else 2

            if aux_op_eigenvalues[_key] is not None:
                result.magnetization.append(aux_op_eigenvalues[_key].real)
            else:
                result.magnetization.append(None)
