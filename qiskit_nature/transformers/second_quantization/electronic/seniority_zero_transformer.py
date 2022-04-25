# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Seniority-Zero Restriction interface."""

from copy import deepcopy
import logging

from functools import partial
from typing import cast, Tuple
import numpy as np

from qiskit_nature import QiskitNatureError, ListOrDictType, settings
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.properties import GroupedProperty, Property
from qiskit_nature.properties.second_quantization import (
    GroupedSecondQuantizedProperty,
    SecondQuantizedProperty,
)
from qiskit_nature.properties.second_quantization.driver_metadata import DriverMetadata
from qiskit_nature.properties.second_quantization.electronic import (
    ParticleNumber,
    ElectronicEnergy,
)
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasisTransform,
    ElectronicBasis,
)
from qiskit_nature.properties.second_quantization.electronic.types import GroupedElectronicProperty
from qiskit_nature.transformers.second_quantization import BaseTransformer

logger = logging.getLogger(__name__)


class SeniorityZeroTransformer(BaseTransformer):
    """The Seniority-Zero restriction."""

    def __init__(self):
        self._num_modes: int = None

    def transform(
        self, grouped_property: GroupedSecondQuantizedProperty
    ) -> GroupedElectronicProperty:
        """Reduces the given `GroupedElectronicProperty` to the restricted Hartree-Fock space.

        Args:
            grouped_property: the `GroupedElectronicProperty` to be transformed.

        Returns:
            A new `GroupedElectronicProperty` instance.

        Raises:
            QiskitNatureError: If the provided `GroupedElectronicProperty` does not contain a
                               `ParticleNumber` instance.
        """
        if not isinstance(grouped_property, GroupedElectronicProperty):
            raise QiskitNatureError(
                "Only `GroupedElectronicProperty` objects can be transformed by this Transformer, "
                f"not objects of type, {type(grouped_property)}."
            )

        electronic_energy = grouped_property.get_property(ElectronicEnergy)
        if electronic_energy is None:
            raise QiskitNatureError(
                "The provided `GroupedElectronicProperty` does not contain a `ElectronicEnergy` "
                "property, which is required by this transformer!"
            )
        electronic_energy = cast(ElectronicEnergy, electronic_energy)

        if not np.allclose(
            electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix(0),
            electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix(1),
        ):
            raise QiskitNatureError(
                "One-body integrals for alpha and beta electrons are not identical, "
                "which is required by this transformer."
            )
        if not np.allclose(
            electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix(0),
            electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix(1),
        ):
            raise QiskitNatureError(
                "Two-body integrals for alpha and beta electrons are not identical, "
                "which is required by this transformer."
            )

        particle_number = grouped_property.get_property(ParticleNumber)
        if particle_number is None:
            raise QiskitNatureError(
                "The provided `GroupedElectronicProperty` does not contain a `ParticleNumber` "
                "property, which is required by this transformer!"
            )
        particle_number = cast(ParticleNumber, particle_number)

        self._num_modes = particle_number.num_spin_orbitals // 2

        grouped_property_transformed = self._transform_property(grouped_property)

        return grouped_property_transformed  # type: ignore[return-value]

    def _transform_property(self, prop: Property) -> Property:
        """Transforms a `Property` object.

        This is a recursive reduction, iterating `GroupedProperty` objects when encountering one.

        Args:
            property: the property object to transform.

        Returns:
            The transformed property object.

        Raises:
            TypeError: if an unexpected Property subtype is encountered.
        """
        transformed_property: Property
        # Code for recursion is copied from ActivateSpaceTransformer
        if isinstance(prop, GroupedProperty):
            transformed_property = deepcopy(prop)

            for internal_property in iter(prop):
                try:
                    transformed_internal_property = self._transform_property(internal_property)
                    if transformed_internal_property is not None:
                        transformed_property.add_property(transformed_internal_property)
                    else:
                        transformed_property.remove_property(internal_property.name)
                except TypeError:
                    logger.warning(
                        "The Property %s of type %s could not be transformed!",
                        internal_property.name,
                        type(internal_property),
                    )
                    continue

            if len(transformed_property._properties) == 0:
                transformed_property = None

        elif isinstance(prop, ElectronicEnergy):
            h_1 = prop.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix()
            h_2 = prop.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix()

            hr_1, hr_2 = self._restricted_electronic_integrals(h_1, h_2)

            transformed_property = ElectronicEnergy.from_raw_integrals(
                ElectronicBasis.MO, hr_1, hr_2
            )
            transformed_property._shift = prop._shift.copy()
            transformed_property.nuclear_repulsion_energy = prop.nuclear_repulsion_energy
            transformed_property.reference_energy = prop.reference_energy

            transformed_property.second_q_ops = partial(  # type: ignore[assignment]
                _second_q_ops_electronic_energy, transformed_property
            )

        elif isinstance(prop, ParticleNumber):
            transformed_property = prop

            transformed_property.second_q_ops = partial(  # type: ignore[assignment]
                _second_q_ops_particle_number, transformed_property
            )
        elif isinstance(prop, DriverMetadata):
            transformed_property = prop
        elif isinstance(prop, SecondQuantizedProperty):
            transformed_property = None
        elif isinstance(prop, ElectronicBasisTransform):
            transformed_property = None

        else:
            raise TypeError(f"{type(prop)} is an unsupported Property-type for this Transformer!")

        return transformed_property

    def _restricted_electronic_integrals(
        self, h_1: np.ndarray, h_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform integrals to the restricted formalism
        arxiv:2002.00035, Appendix A, Equations (A10),(A11),(A12)
        https://arxiv.org/abs/2002.00035

        Args:
            h_1 (np.ndarray): one-body integrals, (N x N)
            h_2 (np.ndarray): two-body integrals, (N x N x N x N)

        Returns:
            hr_1 (np.ndarray): restricted one-body integrals, (N x N)
            hr_2 (np.ndarray): restricted two-body integrals, (N x N)
        """
        # change of notation
        h_2 = np.einsum("ijkl->ljik", h_2)

        hr_1 = np.zeros((self._num_modes, self._num_modes))
        hr_2 = np.zeros((self._num_modes, self._num_modes))

        for i in range(self._num_modes):
            hr_1[i, i] = 2 * h_1[i, i] + h_2[i, i, i, i]

            for j in range(self._num_modes):
                if i != j:
                    hr_1[i, j] = h_2[i, i, j, j]
                    hr_2[i, j] = 2 * h_2[i, j, j, i] - h_2[i, j, i, j]

        return hr_1, hr_2

def _second_q_ops_electronic_energy(self: ElectronicEnergy) -> ListOrDictType[VibrationalOp]:
    """Creates the second quantization operators in the restricted formalism.
    arxiv: 2002.00035, Equation (2)
    https://arxiv.org/abs/2002.00035

    The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

    Args:
        self: Target of monkey-patching.

    Returns:
        A `list` or `dict` of `VibrationalOp` objects.
    """

    hr_1 = self.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix()
    hr_2 = self.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix()

    num_modes = hr_1.shape[0]

    # Functions to create labels for the VibrationalOp. This will prevent the
    # VibrationalOp from triggering a warning about not conserving the number
    # of excitations in each mode (which we violate intentionally).
    label_m = lambda x: "I" * x + "-" + "I" * (num_modes - x - 1)
    label_p = lambda x: "I" * x + "+" + "I" * (num_modes - x - 1)

    # Create pair creation and annihilation operators for the spatial orbitals
    b = [VibrationalOp(label_m(i), num_modes=num_modes, num_modals=1) for i in range(num_modes)]
    b_dag = [VibrationalOp(label_p(i), num_modes=num_modes, num_modals=1) for i in range(num_modes)]

    # TODO: refactor to construct `VibrationalOp` from a list of labels
    # rather than summing multiple instances

    # Build the operators
    op = VibrationalOp(("I" * num_modes, 0.0), num_modes=num_modes, num_modals=1)
    for i in range(num_modes):
        for j in range(num_modes):
            op += b_dag[i] @ b[j] * hr_1[i, j]

            if i != j:
                op += b_dag[i] @ b[i] @ b_dag[j] @ b[j] * hr_2[i, j]
    op = op.simplify()

    if not settings.dict_aux_operators:
        return [op]

    return {self.name: op}


def _second_q_ops_particle_number(self: ParticleNumber) -> ListOrDictType[VibrationalOp]:
    """Creates the second quantized particle number operator in the restricted formalism.

    The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

    Args:
        self: Target of monkey-patching.

    Returns:
        A `list` or `dict` of `VibrationalOp` objects.
    """
    op = VibrationalOp(
        [(f"+_{o}*0 -_{o}*0", 2.0) for o in range(self._num_spin_orbitals // 2)],
        num_modes=self._num_spin_orbitals // 2,
        num_modals=1,
    )

    if not settings.dict_aux_operators:
        return [op]

    return {self.name: op}
