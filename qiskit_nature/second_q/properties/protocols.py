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

"""Some protocols."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import SparseLabelOp


@runtime_checkable
class SparseLabelOpsFactory(Protocol):
    """A protocol indicating :class:`qiskit_nature.second_q.operators.SparseLabelOp` generators."""

    def second_q_ops(self) -> Mapping[str, SparseLabelOp]:
        """Builds the :class:`qiskit_nature.second_q.operators.SparseLabelOp` instances.

        Returns:
            A mapping of strings to `SparseLabelOp` objects.
        """


@runtime_checkable
class Interpretable(Protocol):
    """A protocol determining whether or not an object is interpretable.

    An object is considered interpretable if it implements an `interpret` method.
    """

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in the object's context.

        Args:
            result: the result to add meaning to.
        """
