# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
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

from typing import Mapping

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import FermionicOp


class Magnetization:
    """The Magnetization property.

    The following attributes can be set via the initializer but can also be read and updated once
    the ``Magnetization`` object has been constructed.

    Attributes:
        num_spatial_orbitals (int): the number of spatial orbitals.
    """

    def __init__(self, num_spatial_orbitals: int) -> None:
        """
        Args:

            num_spatial_orbitals: the number of spatial orbitals in the system.
        """
        self.num_spatial_orbitals = num_spatial_orbitals

    def second_q_ops(self) -> Mapping[str, FermionicOp]:
        """Returns the second quantized magnetization operator.

        Returns:
            A mapping of strings to `FermionicOp` objects.
        """
        num_spin_orbitals = 2 * self.num_spatial_orbitals
        op = FermionicOp(
            {
                f"+_{o} -_{o}": 0.5 if o < self.num_spatial_orbitals else -0.5
                for o in range(num_spin_orbitals)
            },
            num_spin_orbitals=num_spin_orbitals,
        )

        return {self.__class__.__name__: op}

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.magnetization = []

        if result.aux_operators_evaluated is None:
            return

        for aux_op_eigenvalues in result.aux_operators_evaluated:
            if not isinstance(aux_op_eigenvalues, dict):
                continue

            _key = self.__class__.__name__

            if aux_op_eigenvalues[_key] is not None:
                result.magnetization.append(aux_op_eigenvalues[_key].real)
            else:
                result.magnetization.append(None)
