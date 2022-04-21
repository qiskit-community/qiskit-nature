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
from qiskit_nature.properties.second_quantization.electronic.electronic_structure_driver_result import (
    ElectronicStructureDriverResult,
)
from qiskit_nature.properties.second_quantization.electronic.types import GroupedElectronicProperty
from qiskit_nature.results.electronic_structure_result import ElectronicStructureResult
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
            transformed_property = prop.__class__()  # type: ignore[call-arg]
            transformed_property.name = prop.name

            if isinstance(prop, ElectronicStructureDriverResult):
                transformed_property.molecule = prop.molecule  # type: ignore[attr-defined]

            for internal_property in iter(prop):
                try:
                    transformed_internal_property = self._transform_property(internal_property)
                    if transformed_internal_property is not None:
                        transformed_property.add_property(transformed_internal_property)
                except TypeError:
                    logger.warning(
                        "The Property %s of type %s could not be transformed!",
                        internal_property.name,
                        type(internal_property),
                    )
                    continue

            # Removing empty GroupedProperty
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
                second_q_ops_electronic_energy, transformed_property
            )

        elif isinstance(prop, ParticleNumber):
            transformed_property = prop

            transformed_property.second_q_ops = partial(  # type: ignore[assignment]
                second_q_ops_particle_number, transformed_property
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
        Transform PyScf integrals to the restricted formalism
        arxiv: 2002.00035, Appendix A, Equations (A10),(A11),(A12)
        https://arxiv.org/pdf/2002.00035.pdf

        Args:
            h_1 (np.ndarray): one-body integrals, (N x N)
            h_2 (np.ndarray): two-body integrals, (N x N x N x N)

        Returns:
            hr_1 (np.ndarray): restricted one-body integrals, (N x N)
            hr_2 (np.ndarray): restricted two-body integrals, (N x N)
        """
        # change of notation (may only be valid for PySCF -> restricted formalism)
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


def second_q_ops_electronic_energy(self: ElectronicEnergy) -> ListOrDictType[VibrationalOp]:
    """Creates the second quantization operators in the restricted formalism
    arxiv: 2002.00035, Equation (2)
    https://arxiv.org/pdf/2002.00035.pdf
    """

    hr_1 = self.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix()
    hr_2 = self.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix()

    num_modes = hr_1.shape[0]

    label_m = lambda x: "I" * x + "-" + "I" * (num_modes - x - 1)
    label_p = lambda x: "I" * x + "+" + "I" * (num_modes - x - 1)

    # Create pair creation and annihilation operators for the spatial orbitals
    b = [VibrationalOp(label_m(i), num_modes=num_modes, num_modals=1) for i in range(num_modes)]
    b_dag = [VibrationalOp(label_p(i), num_modes=num_modes, num_modals=1) for i in range(num_modes)]

    # Build the operators
    op = VibrationalOp(("I" * num_modes, 0.0), num_modes=num_modes, num_modals=1)
    for i in range(num_modes):
        for j in range(num_modes):
            op += b_dag[i] @ b[j] * hr_1[i, j]

            if i != j:
                op += b_dag[i] @ b[i] @ b_dag[j] @ b[j] * hr_2[i, j]
    op = op.reduce()

    if not settings.dict_aux_operators:
        return [op]

    return {self.name: op}


def second_q_ops_particle_number(self) -> ListOrDictType[VibrationalOp]:
    """Returns the second quantized particle number operator.
    The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

    Returns:
        A `list` or `dict` of `VibrationalOp` objects.
    """
    label = lambda x: "I" * x + "N" + "I" * (self._num_spin_orbitals // 2 - x - 1)

    op = VibrationalOp(
        [(label(o), 2.0) for o in range(self._num_spin_orbitals // 2)],
        num_modes=self._num_spin_orbitals // 2,
        num_modals=1,
    )

    if not settings.dict_aux_operators:
        return [op]

    return {self.name: op}
