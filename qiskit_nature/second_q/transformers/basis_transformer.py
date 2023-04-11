# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The basis transformer."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor, Tensor
from qiskit_nature.second_q.problems import BaseProblem, ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    ElectronicDipoleMoment,
    Magnetization,
    ParticleNumber,
)

from .base_transformer import BaseTransformer

LOGGER = logging.getLogger(__name__)


class BasisTransformer(BaseTransformer):
    """A transformer to map from one basis to another.

    Since problems have a basis associated with them (e.g. the
    :class:`qiskit_nature.second_q.problems.ElectronicBasis` in the case of the
    :class:`qiskit_nature.second_q.problems.ElectronicStructureProblem`), this transformer can be
    used to map from one :attr:`initial_basis` to another :attr:`final_basis`.

    For example, this is how you can create an AO-to-MO transformer for an
    :class:`qiskit_nature.second_q.problems.ElectronicStructureProblem`:

    .. code-block:: python

        # assuming you have the transformation coefficients from somewhere
        ao2mo_coeff, ao2mo_coeff_b = ...

        from qiskit_nature.second_q.operators import ElectronicIntegrals
        from qiskit_nature.second_q.problems import ElectronicBasis
        from qiskit_nature.second_q.transformers import BasisTransformer

        transformer = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(ao2mo_coeff, h1_b=ao2mo_coeff_b),
        )

        problem_MO = transformer.transform(problem_AO)

    Attributes:
        initial_basis: the initial basis from which to map away from.
        final_basis: the final basis into which to map into.
        coefficients: the coefficients which transform from the initial to the final basis.
    """

    # TODO: figure out how we can make its interface non-electronic specific
    def __init__(
        self,
        initial_basis: ElectronicBasis,
        final_basis: ElectronicBasis,
        coefficients: PolynomialTensor | ElectronicIntegrals,
    ) -> None:
        """
        Args:
            initial_basis: the initial basis from which to map away from.
            final_basis: the final basis into which to map into.
            coefficients: the coefficients which transform from the initial to the final basis.
        """
        self.initial_basis = initial_basis
        self.final_basis = final_basis
        self.coefficients = coefficients

    def invert(self) -> BasisTransformer:
        """Invert the transformer to do the reversed transformation.

        Returns:
            A new ``BasisTransformer`` mapping from :attr:`final_basis` to :attr:`initial_basis`.
        """
        return BasisTransformer(
            self.final_basis,
            self.initial_basis,
            self.coefficients.__class__.apply(np.transpose, self.coefficients, validate=False),
        )

    def transform(self, problem: BaseProblem) -> BaseProblem:
        if isinstance(problem, ElectronicStructureProblem):
            return self._transform_electronic_structure_problem(problem)
        else:
            raise NotImplementedError(
                f"The problem of type, {type(problem)}, is not supported by this transformer."
            )

    def _transform_electronic_structure_problem(
        self, problem: ElectronicStructureProblem
    ) -> ElectronicStructureProblem:
        """Transforms an :class:`qiskit_nature.second_q.problems.ElectronicStructureProblem`.

        .. note::
            This function will log warnings, when encountering unsupported/unknown properties in the
            :class:`qiskit_nature.second_q.problems.ElectronicPropertiesContainer`.

        Args:
            problem: the ``ElectronicStructureProblem`` to transform.

        Raises:
            QiskitNatureError: if the basis of the supplied `ElectronicStructureProblem` does not
                match the :attr:`initial_basis`.

        Returns:
            The transformed ``ElectronicStructureProblem``.
        """
        if problem.basis != self.initial_basis:
            raise QiskitNatureError(
                f"The problems' basis, {problem.basis}, does not match the initial basis of this "
                f"transformer, {self.initial_basis}."
            )

        new_problem = ElectronicStructureProblem(
            cast(ElectronicEnergy, self.transform_hamiltonian(problem.hamiltonian))
        )
        new_problem.basis = self.final_basis
        new_problem.molecule = problem.molecule
        new_problem.reference_energy = problem.reference_energy
        new_problem.num_particles = problem.num_particles
        new_problem.num_spatial_orbitals = problem.num_spatial_orbitals

        for prop in problem.properties:
            if isinstance(prop, ElectronicDipoleMoment):
                new_problem.properties.electronic_dipole_moment = (
                    self._transform_electronic_dipole_moment(prop)
                )
            elif isinstance(prop, ElectronicDensity):
                new_problem.properties.electronic_density = self.transform_electronic_integrals(
                    prop
                )
            elif isinstance(prop, (AngularMomentum, Magnetization, ParticleNumber)):
                new_problem.properties.add(prop)
            else:
                LOGGER.warning("Encountered an unsupported property of type '%s'.", type(prop))

        return new_problem

    def transform_electronic_integrals(self, integrals: ElectronicIntegrals) -> ElectronicIntegrals:
        """Transforms an :class:`qiskit_nature.second_q.operators.ElectronicIntegrals` instance.

        Args:
            integrals: the ``ElectronicIntegrals`` to transform.

        Raises:
            QiskitNatureError: when using this method on a ``BasisTransformer`` that does not store
                its :attr:`coefficients` as ``ElectronicIntegrals``, too.

        Returns:
            The transformed ``ElectronicIntegrals``.
        """
        if not isinstance(self.coefficients, ElectronicIntegrals):
            raise QiskitNatureError(
                "You cannot transform ElectronicIntegrals with a BasisTransformer containing "
                f"coefficients of type, {type(self.coefficients)}, rather than ElectronicIntegrals."
            )

        prsq = "prsq"
        iklj = "iklj"

        two_body_aa = integrals.alpha.get("++--", None)
        if two_body_aa is not None:
            # TODO: remove extra-wrapping of Tensor once settings.tensor_unwrapping is removed
            if not isinstance(two_body_aa, Tensor):
                two_body_aa = Tensor(two_body_aa)

            prsq = "".join(two_body_aa._reverse_label_template(prsq))
            iklj = "".join(two_body_aa._reverse_label_template(iklj))

        einsum_map = {
            "jk,ji,kl->il": ("+-",) * 4,
            f"{prsq},pi,qj,rk,sl->{iklj}": ("++--", *("+-",) * 4, "++--"),
        }

        transformed_integrals = ElectronicIntegrals.einsum(
            einsum_map, integrals, *(self.coefficients,) * 4
        )

        if not self.coefficients.beta.is_empty() and transformed_integrals.beta_alpha.is_empty():
            transformed_integrals.beta_alpha = PolynomialTensor.einsum(
                {f"{prsq},pi,qj,rk,sl->{iklj}": ("++--", *("+-",) * 4, "++--")},
                integrals.alpha if integrals.beta_alpha.is_empty() else integrals.beta_alpha,
                *(self.coefficients.beta,) * 2,
                *(self.coefficients.alpha,) * 2,
            )

        return transformed_integrals

    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        if isinstance(hamiltonian, ElectronicEnergy):
            integrals = hamiltonian.electronic_integrals
            hamiltonian.electronic_integrals = self.transform_electronic_integrals(integrals)
            return hamiltonian
        else:
            raise NotImplementedError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "transformer."
            )

    def _transform_electronic_dipole_moment(
        self, dipole_moment: ElectronicDipoleMoment
    ) -> ElectronicDipoleMoment:
        """Transforms an :class:`qiskit_nature.second_q.properties.ElectronicDipoleMoment`.

        Args:
            dipole_moment: the ``ElectronicDipoleMoment`` to transform.

        Returns:
            The transformed ``ElectronicDipoleMoment``.
        """
        new_dipole_moment = ElectronicDipoleMoment(
            self.transform_electronic_integrals(dipole_moment.x_dipole),
            self.transform_electronic_integrals(dipole_moment.y_dipole),
            self.transform_electronic_integrals(dipole_moment.z_dipole),
        )
        new_dipole_moment.reverse_dipole_sign = dipole_moment.reverse_dipole_sign
        new_dipole_moment.nuclear_dipole_moment = dipole_moment.nuclear_dipole_moment
        return new_dipole_moment
