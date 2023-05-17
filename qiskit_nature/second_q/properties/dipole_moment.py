# This code is part of Qiskit.
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

"""The ElectronicDipoleMoment property."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, Tuple, cast

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp


# A dipole moment, when present as X, Y and Z components will normally have float values for all the
# components. However when using Z2Symmetries, if the dipole component operator does not commute
# with the symmetry then no evaluation is done and None will be used as the 'value' indicating no
# measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class ElectronicDipoleMoment:
    r"""The ElectronicDipoleMoment property.

    This Property implements the operator which evaluates the **electronic** dipole moment, based on
    the electronic integrals:

    .. math::
        \vec{d}_{pq} = \int \phi_{p} \vec{r} \phi_{q} ,

    where the integral is over the all space. The operator can then be expressed as:

    .. math::
        \hat{d} = \sum_{p, q} d^x_{pq} a^\dagger_p a_q ,

    where :math:`x` can be any of the three Cartesian axes (:math:`\{x, y, z\}`).

    Just like in the :class:`qiskit_nature.second_q.hamiltonians.ElectronicEnergy`, the nuclear
    contribution is stored **separately** in the :attr:`nuclear_dipole_moment` attribute and will
    **not** be included into the generated operator. It is however possible, to manually add this
    constant shift to the electronic dipole components as a constant term in the internal
    :class:`qiskit_nature.second_q.operators.ElectronicIntegrals` instances. Assuming you have
    obtained an ``ElectronicDipoleMoment`` instance (for example from one of the classical code
    drivers), you can add the nuclear component to be included in the qubit operator like so:

    .. code-block:: python

        from qiskit_nature.second_q.operators import PolynomialTensor

        # you have obtained your dipole moment property and store it in this variable
        dipole: ElectronicDipoleMoment
        nuclear_dip = dipole.nuclear_dipole_moment
        dipole.x_dipole.alpha += PolynomialTensor({"": nuclear_dip[0]})
        dipole.y_dipole.alpha += PolynomialTensor({"": nuclear_dip[1]})
        dipole.z_dipole.alpha += PolynomialTensor({"": nuclear_dip[2]})

    The following attributes can be set via the initializer but can also be read and updated once
    the ``ElectronicDipoleMoment`` object has been constructed.

    Attributes:
        x_dipole (ElectronicIntegrals): the ``ElectronicIntegrals`` for the :math:`x`-axis component.
        y_dipole (ElectronicIntegrals): the ``ElectronicIntegrals`` for the :math:`y`-axis component.
        z_dipole (ElectronicIntegrals): the ``ElectronicIntegrals`` for the :math:`z`-axis component.
        constants (MutableMapping[str, DipoleTuple]): a mapping of constant dipole offsets, not
            mapped to the qubit operator. Each entry must be a tuple of length three (for the three
            Cartesian axes).
        reverse_dipole_sign: whether or not to reverse the sign of the computed electronic dipole
            moment when adding it to the :attr:`nuclear_dipole_moment` to obtain the total.
    """

    def __init__(
        self,
        x_dipole: ElectronicIntegrals,
        y_dipole: ElectronicIntegrals,
        z_dipole: ElectronicIntegrals,
        *,
        constants: MutableMapping[str, DipoleTuple] = None,
        reverse_dipole_sign: bool = False,
    ) -> None:
        """
        Args:
            x_dipole: the :class:`qiskit_nature.second_q.operators.ElectronicIntegrals` for the
                :math:`x`-axis component.
            y_dipole: the :class:`qiskit_nature.second_q.operators.ElectronicIntegrals` for the
                :math:`y`-axis component.
            z_dipole: the :class:`qiskit_nature.second_q.operators.ElectronicIntegrals` for the
                :math:`z`-axis component.
            constants: a mapping of constant dipole offsets, not mapped to the qubit operator.
                Each entry must be a tuple of length three (for the three Cartesian axes).
            reverse_dipole_sign: whether or not to reverse the dipole sign.
        """
        self.x_dipole = x_dipole
        self.y_dipole = y_dipole
        self.z_dipole = z_dipole
        self.constants = constants if constants is not None else {}
        self.reverse_dipole_sign = reverse_dipole_sign

    @property
    def nuclear_dipole_moment(self) -> DipoleTuple | None:
        """The nuclear dipole moment.

        See :attr:`qiskit_nature.second_q.hamiltonians.ElectronicEnergy.nuclear_repulsion_energy`
        for an example on how to add the constant terms as offsets to the internal
        :class:`qiskit_nature.second_q.operators.ElectronicIntegrals`.
        """
        return self.constants.get("nuclear_dipole_moment", None)

    @nuclear_dipole_moment.setter
    def nuclear_dipole_moment(self, d_nuc: DipoleTuple) -> None:
        self.constants["nuclear_dipole_moment"] = d_nuc

    def second_q_ops(self) -> Mapping[str, FermionicOp]:
        """Returns the second quantized dipole moment operators.

        Returns:
            A mapping of strings to `FermionicOp` objects.
        """
        ops = {}
        ops["XDipole"] = FermionicOp.from_polynomial_tensor(self.x_dipole.second_q_coeffs())
        ops["YDipole"] = FermionicOp.from_polynomial_tensor(self.y_dipole.second_q_coeffs())
        ops["ZDipole"] = FermionicOp.from_polynomial_tensor(self.z_dipole.second_q_coeffs())
        return ops

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`qiskit_nature.second_q.problems.EigenstateResult`.

        In particular, this extracts the evaluated electronic dipole moment values from the
        auxiliary operator eigenvalues and this adds the constant dipole shifts stored in this
        property to the result object.

        Args:
            result: the result to add meaning to.
        """
        result.nuclear_dipole_moment = self.nuclear_dipole_moment
        result.reverse_dipole_sign = self.reverse_dipole_sign
        result.computed_dipole_moment = []
        result.extracted_transformer_dipoles = []

        if result.aux_operators_evaluated is None:
            return

        for aux_op_eigenvalues in result.aux_operators_evaluated:
            if not isinstance(aux_op_eigenvalues, dict):
                continue

            dipole_moment = [None] * 3
            for idx, axis in enumerate("XYZ"):
                moment = aux_op_eigenvalues.get(f"{axis}Dipole", None)
                if moment is not None:
                    dipole_moment[idx] = moment.real

            result.computed_dipole_moment.append(cast(DipoleTuple, tuple(dipole_moment)))

            result.extracted_transformer_dipoles.append(
                {
                    name: constant
                    for name, constant in self.constants.items()
                    if name != "nuclear_dipole_moment"
                }
            )
