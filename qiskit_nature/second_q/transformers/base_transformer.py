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

"""The Base Operator Transformer interface."""

from abc import ABC, abstractmethod

from qiskit_nature.second_q.operator_factories import GroupedSecondQuantizedProperty


class BaseTransformer(ABC):
    """The interface for implementing methods which map from one
    :class:`~qiskit_nature.properties.GroupedProperty` to another.
    These methods may or may not affect the size of the Hilbert space.
    """

    @abstractmethod
    def transform(
        self, grouped_property: GroupedSecondQuantizedProperty
    ) -> GroupedSecondQuantizedProperty:
        """Transforms one :class:`~qiskit_nature.properties.GroupedProperty` into another one.
        This may or may not affect the size of the Hilbert space.

        Args:
            grouped_property: the `GroupedProperty` to be transformed.

        Returns:
            A new `GroupedProperty` instance.
        """
        raise NotImplementedError()
