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

"""The VibrationalEnergy property."""

from __future__ import annotations

from typing import cast, Generator, Optional

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import VibrationalOp

from qiskit_nature.second_q.properties.bases import VibrationalBasis
from qiskit_nature.second_q.properties.integrals import VibrationalIntegrals

from .hamiltonian import Hamiltonian


class VibrationalEnergy(Hamiltonian):
    """The VibrationalEnergy property.

    This is the main property of any vibrational structure problem. It constructs the Hamiltonian
    whose eigenvalue is the target of a later used Quantum algorithm.
    """

    def __init__(
        self,
        vibrational_integrals: list[VibrationalIntegrals],
        truncation_order: Optional[int] = None,
        basis: Optional[VibrationalBasis] = None,
    ) -> None:
        # pylint: disable=line-too-long
        """
        Args:
            vibrational_integrals: a list of
                :class:`~qiskit_nature.second_q.properties.integrals.VibrationalIntegrals`.
            truncation_order: an optional truncation order for the highest number of body terms to
                include in the constructed Hamiltonian.
            basis: the
                :class:`~qiskit_nature.second_q.properties.bases.VibrationalBasis`
                through which to map the integrals into second quantization. This attribute **MUST**
                be set before the second-quantized operator can be constructed.
        """
        self._vibrational_integrals: dict[int, VibrationalIntegrals] = {}
        for integral in vibrational_integrals:
            self.add_vibrational_integral(integral)
        self._truncation_order = truncation_order
        self._basis: VibrationalBasis = basis

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @property
    def truncation_order(self) -> int:
        """Returns the truncation order."""
        return self._truncation_order

    @truncation_order.setter
    def truncation_order(self, truncation_order: int) -> None:
        """Sets the truncation order."""
        self._truncation_order = truncation_order

    def __str__(self) -> str:
        string = ["VibrationalEnergy:"]
        string += [f"\t{line}" for line in str(self.basis).split("\n")]
        for ints in self._vibrational_integrals.values():
            string += [f"\t{ints}"]
        return "\n".join(string)

    def __iter__(self) -> Generator[VibrationalIntegrals, None, None]:
        """Returns the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[VibrationalIntegrals, None, None]:
        """A generator-iterator method [1] iterating over all internal ``VibrationalIntegrals``.

        [1]: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
        """
        for ints in self._vibrational_integrals.values():
            yield ints

    def add_vibrational_integral(self, integral: VibrationalIntegrals) -> None:
        # pylint: disable=line-too-long
        """Adds a
        :class:`~qiskit_nature.second_q.properties.integrals.VibrationalIntegrals`
        instance to the internal storage.

        Internally, the
        :class:`~qiskit_nature.second_q.properties.integrals.VibrationalIntegrals`
        are stored in a dictionary sorted by their number of body terms. This simplifies access
        based on these properties (see ``get_vibrational_integral``) and avoids duplicate,
        inconsistent entries.

        Args:
            integral: the
                :class:`~qiskit_nature.second_q.properties.integrals.VibrationalIntegrals`
                to add.
        """
        self._vibrational_integrals[integral._num_body_terms] = integral

    def get_vibrational_integral(self, num_body_terms: int) -> Optional[VibrationalIntegrals]:
        """Gets an
        :class:`~qiskit_nature.second_q.properties.integrals.VibrationalIntegrals`
        given the number of body terms.

        Args:
            num_body_terms: the number of body terms of the queried integrals.

        Returns:
            The queried integrals object (or None if unavailable).
        """
        return self._vibrational_integrals.get(num_body_terms, None)

    @property
    def register_length(self) -> int:
        return sum(self.basis._num_modals_per_mode)

    def second_q_op(self) -> VibrationalOp:
        """Returns the second quantized vibrational energy operator.

        Returns:
            A `dict` of `VibrationalOp` objects.
        """
        ops = []
        for num_body, ints in self._vibrational_integrals.items():
            if self._truncation_order is not None and num_body > self._truncation_order:
                break
            ints.basis = self.basis
            ops.append(ints.to_second_q_op())

        return cast(VibrationalOp, sum(ops))

    def interpret(
        self, result: "qiskit_nature.second_q.problemsEigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
