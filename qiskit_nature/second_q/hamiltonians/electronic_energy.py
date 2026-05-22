# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ElectronicEnergy property."""

from __future__ import annotations

from copy import copy
from typing import MutableMapping

import numpy as np

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp, PolynomialTensor

from .hamiltonian import Hamiltonian


class ElectronicEnergy(Hamiltonian):
    r"""The electronic energy Hamiltonian.

    This class implements the following Hamiltonian:

    .. math::
        \sum_{p, q} h_{pq} a^\dagger_p a_q
        + \sum_{p, q, r, s} g_{pqrs} a^\dagger_p a^\dagger_q a_r a_s ,

    where :math:`h_{pq}` and :math:`g_{pqrs}` are the one- and two-body electronic integrals,
    stored in an :class:`~qiskit_nature.second_q.operators.ElectronicIntegrals` container.
    When dealing with separate coefficients for the :math:`\alpha`- and :math:`\beta`-spin electrons,
    the unrestricted-spin Hamiltonian can be obtained from the one above in a straight-forward
    manner, following any quantum chemistry textbook.

    You can construct an instance of this Hamiltonian in multiple ways:

    1. With an existing instance of :class:`~qiskit_nature.second_q.operators.ElectronicIntegrals`:

    .. code-block:: python

        integrals: ElectronicIntegrals = ...

        from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

        hamiltonian = ElectronicEnergy(integrals, constants={"nuclear_repulsion_energy": 1.0})

    2. From a raw set of integral coefficient matrices:

    .. code-block:: python

        # assuming, you have your one- and two-body integrals from somewhere
        h1_a, h2_aa, h1_b, h2_bb, h2_ba = ...

        hamiltonian = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa, h1_b, h2_bb, h2_ba)
        hamiltonian.nuclear_repulsion_energy = 1.0

    Note, how we specified the nuclear repulsion energy as a constant energy offset in the above
    examples. This term will not be included in the mapped qubit operator since it is a constant
    offset term and does not need to incur any errors from being measured on a quantum device.
    It is however possible to include constant energy terms inside of the
    :class:`~qiskit_nature.second_q.operators.ElectronicIntegrals` container, if you want it to be
    included in the qubit operator, once mapping the second-quantized operator to the qubit space
    (see also :class:`~qiskit_nature.second_q.mappers.QubitMapper`).

    .. code-block:: python

        from qiskit_nature.second_q.operators import PolynomialTensor

        e_nuc = hamiltonian.nuclear_repulsion_energy
        hamiltonian.electronic_integrals.alpha += PolynomialTensor({"": e_nuc})
        hamiltonian.nuclear_repulsion_energy = None

    It is also possible to add other constant energy offsets to the :attr:`.constants` attribute of
    this Hamiltonian. All offsets registered in that dictionary will **not** be mapped to the qubit
    operator.

    .. code-block:: python

        hamiltonian.constants["my custom offset"] = 5.0

        # be careful, the following overwrites the hamiltonian.nuclear_repulsion_energy value
        hamiltonian.constants["nuclear_repulsion_energy"] = 10.0

    Attributes:
        electronic_integrals: The :class:`qiskit_nature.second_q.operators.ElectronicIntegrals`.
        constants: A mapping of constant energy offsets, not mapped to the qubit operator.
    """

    def __init__(
        self,
        electronic_integrals: ElectronicIntegrals,
        *,
        constants: MutableMapping[str, float] = None,
    ) -> None:
        """
        Args:
            electronic_integrals: The container with the one- and two-body coefficients.
            constants: A mapping of constant energy offsets.
        """
        self.electronic_integrals = electronic_integrals
        self.constants = constants if constants is not None else {}

    @property
    def register_length(self) -> int | None:
        return self.electronic_integrals.register_length

    @property
    def nuclear_repulsion_energy(self) -> float | None:
        """The nuclear repulsion energy.

        This constant energy offset does **not** get included in the generated operator.
        Add it as a constant term to the :attr:`electronic_integrals` and remove it here, if you
        want to include it in the generated operator:

        .. code-block:: python

            from qiskit_nature.second_q.operators import PolynomialTensor

            hamiltonian = ElectronicEnergy(...)
            hamiltonian.electronic_integrals.alpha += PolynomialTensor({
                "": hamiltonian.nuclear_repulsion_energy
            })
            hamiltonian.nuclear_repulsion_energy = None
        """
        return self.constants.get("nuclear_repulsion_energy", None)

    @nuclear_repulsion_energy.setter
    def nuclear_repulsion_energy(self, e_nuc: float | None) -> None:
        if e_nuc is None:
            self.constants.pop("nuclear_repulsion_energy")
        else:
            self.constants["nuclear_repulsion_energy"] = e_nuc

    # pylint: disable=invalid-name, disable=too-many-positional-arguments
    @classmethod
    def from_raw_integrals(
        cls,
        h1_a: np.ndarray,
        h2_aa: np.ndarray,
        h1_b: np.ndarray | None = None,
        h2_bb: np.ndarray | None = None,
        h2_ba: np.ndarray | None = None,
        *,
        validate: bool = True,
        auto_index_order: bool = True,
    ) -> ElectronicEnergy:
        """Constructs a hamiltonian instance from raw integrals.

        This function simply calls
        :meth:`~qiskit_nature.second_q.operators.ElectronicIntegrals.from_raw_integrals`.
        See its documentation for more details.

        Args:
            h1_a: the alpha-spin one-body coefficients.
            h2_aa: the alpha-alpha-spin two-body coefficients.
            h1_b: the beta-spin one-body coefficients.
            h2_bb: the beta-beta-spin two-body coefficients.
            h2_ba: the beta-alpha-spin two-body coefficients.
            validate: whether or not to validate the coefficient matrices.
            auto_index_order: whether or not to automatically convert the matrices to physicists'
                order.

        Returns:
            The resulting ``ElectronicEnergy`` instance.
        """
        return cls(
            ElectronicIntegrals.from_raw_integrals(
                h1_a,
                h2_aa,
                h1_b,
                h2_bb,
                h2_ba,
                validate=validate,
                auto_index_order=auto_index_order,
            )
        )

    def second_q_op(self) -> FermionicOp:
        """Returns the second quantized operator constructed from the contained electronic integrals.

        Returns:
            A ``FermionicOp`` instance.
        """
        return FermionicOp.from_polynomial_tensor(self.electronic_integrals.second_q_coeffs())

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`.

        In particular, this adds the constant energy shifts stored in this hamiltonian to the result
        object.

        Args:
            result: the result to add meaning to.
        """
        result.extracted_transformer_energies = copy(self.constants)
        result.nuclear_repulsion_energy = result.extracted_transformer_energies.pop(
            "nuclear_repulsion_energy", None
        )

    def coulomb(self, density: ElectronicIntegrals) -> ElectronicIntegrals:
        r"""Computes the Coulomb term for the given reduced density matrix.

        .. math::
            J_{qr} = \sum g_{pqrs} D_{ps}

        Args:
            density: the reduced density matrix.

        Returns:
            The Coulomb operator coefficients.

        Raises:
            NotImplementedError: when encountering :class:`.SymmetricTwoBodyIntegrals` inside of
                :attr:`.ElectronicEnergy.electronic_integrals`.
        """
        two_body_aa = self.electronic_integrals.alpha.get("++--", None)

        einsum = f"{''.join(two_body_aa._reverse_label_template('pqrs'))},ps->qr"
        coulomb = ElectronicIntegrals.einsum(
            {einsum: ("++--", "+-", "+-")}, self.electronic_integrals, density
        )

        if self.electronic_integrals.beta_alpha.is_empty() and density.beta.is_empty():
            coulomb *= 2.0  # type: ignore
        else:
            if self.electronic_integrals.beta_alpha.is_empty():
                beta_alpha = self.electronic_integrals.two_body.alpha
            else:
                beta_alpha = self.electronic_integrals.beta_alpha
            coulomb.alpha += PolynomialTensor.einsum(
                {einsum: ("++--", "+-", "+-")}, beta_alpha, density.beta
            )
            einsum = einsum[2:4] + einsum[:2] + einsum[4:]
            coulomb.beta += PolynomialTensor.einsum(
                {einsum: ("++--", "+-", "+-")}, beta_alpha, density.alpha
            )

        return coulomb

    def exchange(self, density: ElectronicIntegrals) -> ElectronicIntegrals:
        r"""Computes the Exchange term for the given reduced density matrix.

        .. math::
            K_{pr} = \sum g_{pqrs} D_{qs}

        Args:
            density: the reduced density matrix.

        Returns:
            The Exchange operator coefficients.

        Raises:
            NotImplementedError: when encountering :class:`.SymmetricTwoBodyIntegrals` inside of
                :attr:`.ElectronicEnergy.electronic_integrals`.
        """
        two_body_aa = self.electronic_integrals.alpha.get("++--", None)

        einsum = f"{''.join(two_body_aa._reverse_label_template('pqrs'))},qs->pr"
        exchange = ElectronicIntegrals.einsum(
            {einsum: ("++--", "+-", "+-")}, self.electronic_integrals, density
        )
        return exchange

    def fock(self, density: ElectronicIntegrals) -> ElectronicIntegrals:
        r"""Computes the Fock operator for the given reduced density matrix.

        .. math::
            F_{pq} = h_{pq} + J_{pq} - K_{pq}

        where :math:`J` and :math:`K` are the :meth:`coulomb` and :meth:`exchange` terms,
        respectively.

        Args:
            density: the reduced density matrix.

        Returns:
            The Fock operator coefficients.
        """
        return self.electronic_integrals.one_body + self.coulomb(density) - self.exchange(density)
