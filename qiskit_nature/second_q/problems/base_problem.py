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
# This code is part of Qiskit.
"""The Base Problem class."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit.opflow import PauliSumOp, Z2Symmetries

from qiskit_nature import ListOrDictType
from qiskit_nature.second_q.operators import QubitConverter
from qiskit_nature.deprecation import DeprecatedType, deprecate_property
from qiskit_nature.second_q.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.second_q.drivers.second_quantization import BaseDriver
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.operator_factories import GroupedSecondQuantizedProperty

from .base_transformer import BaseTransformer
from .eigenstate_result import EigenstateResult

LegacyDriverResult = Union[QMolecule, WatsonHamiltonian]


class BaseProblem(ABC):
    """Base Problem"""

    def __init__(
        self,
        driver: Optional[BaseDriver] = None,
        transformers: Optional[List[BaseTransformer]] = None,
        main_property_name: str = "",
    ):
        """

        Args:
            driver: A driver encoding the molecule information.
            transformers: A list of transformations to be applied to the driver result.
            main_property_name: A main property name for the problem
        """

        self.driver = driver
        self.transformers = transformers or []

        self._molecule_data: Optional[LegacyDriverResult] = None
        self._molecule_data_transformed: Optional[LegacyDriverResult] = None

        self._grouped_property: Optional[GroupedSecondQuantizedProperty] = None
        self._grouped_property_transformed: Optional[GroupedSecondQuantizedProperty] = None

        self._main_property_name: str = main_property_name

    @property  # type: ignore[misc]
    @deprecate_property(
        "0.2.0",
        new_type=DeprecatedType.PROPERTY,
        new_name="grouped_property",
    )
    def molecule_data(self) -> Optional[LegacyDriverResult]:
        """Returns the raw molecule data object."""
        return self._molecule_data

    @property  # type: ignore[misc]
    @deprecate_property(
        "0.2.0",
        new_type=DeprecatedType.PROPERTY,
        new_name="grouped_property_transformed",
    )
    def molecule_data_transformed(self) -> Optional[LegacyDriverResult]:
        """Returns the raw transformed molecule data object."""
        return self._molecule_data_transformed

    @property
    def grouped_property(self) -> Optional[GroupedSecondQuantizedProperty]:
        """Returns the
        :class:`~qiskit_nature.second_q.operator_factories.GroupedSecondQuantizedProperty`
        object."""
        return self._grouped_property

    @property
    def grouped_property_transformed(self) -> Optional[GroupedSecondQuantizedProperty]:
        """Returns the transformed
        :class:`~qiskit_nature.second_q.operator_factories.GroupedSecondQuantizedProperty`
        object."""
        return self._grouped_property_transformed

    @property
    def main_property_name(self) -> str:
        """Returns the name of the property producing the main operator."""
        return self._main_property_name

    @property
    def num_particles(self) -> Optional[Tuple[int, int]]:
        """Returns the number of particles, if available."""
        return None

    @abstractmethod
    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:
        """Returns the second quantized operators associated with this Property.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """
        raise NotImplementedError()

    def _transform(self, data):
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data

    def symmetry_sector_locator(
        self,
        z2_symmetries: Z2Symmetries,
        converter: QubitConverter,
    ) -> Optional[List[int]]:
        # pylint: disable=unused-argument
        """Given the detected Z2Symmetries, it can determine the correct sector of the tapered
        operators so the correct one can be returned

        Args:
            z2_symmetries: the z2 symmetries object.
            converter: the qubit converter instance used for the operator conversion that
                symmetries are to be determined for.

        Returns:
            the sector of the tapered operators with the problem solution
        """
        return None

    @abstractmethod
    def interpret(self, raw_result: EigenstateResult) -> EigenstateResult:
        """Interprets an EigenstateResult in the context of this problem.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An interpreted `EigenstateResult` in the form of a subclass of it. The actual type
            depends on the problem that implements this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """
        raise NotImplementedError()

    @abstractmethod
    def hopping_qeom_ops(
        self,
        qubit_converter: QubitConverter,
        excitations: Union[
            str,
            int,
            List[int],
            Callable[[int, Tuple[int, int]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
        ] = "sd",
    ) -> Optional[
        Tuple[
            Dict[str, PauliSumOp],
            Dict[str, List[bool]],
            Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
        ]
    ]:
        """Generates the hopping operators and their commutativity information for the specified set
        of excitations.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simple cases for this input are

                :`str`: containing any of the following characters: `s`, `d`, `t` or `q`.
                :`int`: a single, positive integer denoting the excitation type (1 == `s`, etc.).
                :`List[int]`: a list of positive integers.
                :`Callable`: a function which is used to generate the excitations.
                    For more details on how to write such a function refer to one of the default
                    methods, :meth:`generate_fermionic_excitations` or
                    :meth:`generate_vibrational_excitations`.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """
        raise NotImplementedError()
