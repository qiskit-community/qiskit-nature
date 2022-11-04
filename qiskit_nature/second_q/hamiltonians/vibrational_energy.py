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

"""The vibrational energy Hamiltonian."""

from __future__ import annotations

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import VibrationalIntegrals, VibrationalOp

from .hamiltonian import Hamiltonian


class VibrationalEnergy(Hamiltonian):
    r"""The vibrational energy Hamiltonian.

    This class implements the following Hamiltonian:

    .. math::
        \sum_{l=1}^L \sum_{k_l,h_l}^{N_l}
            \langle \phi_{k_l} | T(Q_l) + V^{[l]}(Q_l) | \phi_{h_l} \rangle a^\dagger_{k_l} a_{h_l}
        + \sum_{l<m}^L \sum_{k_l,h_l}^{N_l} \sum_{k_m,h_m}^{N_m}
            \langle \phi_{k_l} \phi_{k_m} | V^{[l,m]}(Q_l, Q_m) | \phi_{h_l} \phi_{h_m} \rangle
            a^\dagger_{k_l} a^\dagger_{k_m} a_{h_l} a_{h_m}
        + \ldots

    where :math:`Q` denotes a vibrational mode, :math:`T` denotes the kinetic term, and :math:`V`
    denotes the potential terms acting on multiple modes. The subscripts :math:`k` and :math:`h` are
    indexing the modals which each mode :math:`l` gets expanded into.

    For a detailed explanation please refer to reference [1].

    The following attributes can be set via the initializer but can also be read and updated once
    the ``VibrationalEnergy`` object has been constructed.

    Attributes:
        vibrational_integrals (VibrationalIntegrals): the integral coefficients.
        truncation_order (int | None): the maximum order of multi-body terms to include in the
            operator.

    References:
        [1]: P. Ollitrault et al. `arXiv:2003.12578 <https://arxiv.org/abs/2003.12578>`_.
    """

    def __init__(
        self,
        vibrational_integrals: VibrationalIntegrals,
        *,
        truncation_order: int | None = None,
    ) -> None:
        """
        Args:
            vibrational_integrals: the container with the integral coefficients.
            truncation_order: the maximum order of multi-body terms to include in the operator.
        """
        self.vibrational_integrals = vibrational_integrals
        self.truncation_order = truncation_order

    @property
    def register_length(self) -> int | None:
        return None

    @classmethod
    def from_raw_integrals(cls, integrals: dict[tuple[int, ...], complex]) -> VibrationalEnergy:
        """Constructs a hamiltonian instance from raw integrals.

        This function simply calls
        :meth:`qiskit_nature.second_q.operators.VibrationalIntegrals.from_raw_integrals`.
        See its documentation for more details.

        Args:
            integrals: a mapping of matrix index tuples to coefficients.

        Returns:
            The resulting ``VibrationalEnergy`` instance.
        """
        return cls(VibrationalIntegrals.from_raw_integrals(integrals))

    def second_q_op(self) -> VibrationalOp:
        """Returns the second quantized vibrational energy operator.

        Returns:
            A ``dict`` of ``VibrationalOp`` objects.
        """
        truncated_integrals = self.vibrational_integrals
        if self.truncation_order is not None:
            truncated_integrals = VibrationalIntegrals(
                {
                    key: value
                    for key, value in self.vibrational_integrals.items()
                    if len(key) <= 3 * self.truncation_order
                },
                validate=False,
            )
        return VibrationalOp.from_polynomial_tensor(truncated_integrals)

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`.

        Args:
            result: The result to add meaning to.
        """
