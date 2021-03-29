# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.opflow import PauliSumOp

from qiskit_nature.drivers import BaseDriver, QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.results import EigenstateResult
from qiskit_nature.transformers import BaseTransformer


class BaseProblem(ABC):
    """Base Problem"""

    def __init__(self, driver: BaseDriver,
                 transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            driver: A driver encoding the molecule information.
            transformers: A list of transformations to be applied to the molecule.
        """

        self.driver = driver
        self.transformers = transformers or []

        self._molecule_data: Union[QMolecule, WatsonHamiltonian] = None
        self._molecule_data_transformed: Union[QMolecule, WatsonHamiltonian] = None

    @property
    def molecule_data(self) -> Union[QMolecule, WatsonHamiltonian]:
        """Returns the raw molecule data object."""
        return self._molecule_data

    @property
    def molecule_data_transformed(self) -> Union[QMolecule, WatsonHamiltonian]:
        """Returns the raw transformed molecule data object."""
        return self._molecule_data_transformed

    @abstractmethod
    def second_q_ops(self):
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: ... .
        """
        raise NotImplementedError()

    def _transform(self, data):
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data

    @abstractmethod
    def interpret(self, raw_mes_result: EigenstateResult):
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_mes_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_default_filter_criterion(self) -> Optional[Callable[[Union[List, np.ndarray], float,
                                                                 Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        raise NotImplementedError()

    @abstractmethod
    def hopping_ops(self, qubit_converter: QubitConverter,
                    excitations: Union[str, int, List[int],
                                       Callable[[int, Tuple[int, int]],
                                                List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
                                       ] = 'sd',
                    ) -> Tuple[Dict[str, PauliSumOp], Dict[str, List[bool]],
                               Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        """Generates the hopping operators and their commutativity information for the specified set
        of excitations.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simples cases for this input are:
                - a `str` containing any of the following charactes: `s`, `d`, `t` or `q`.
                - a single, positive `int` denoting the exitation type (1 == `s`, etc.).
                - a list of positive integers.
                - and finally a callable which can be used to specify a custom list of excitations.
                  For more details on how to write such a function refer to one of the default
                  methods, :meth:`generate_fermionic_excitations` or
                  :meth:`generate_vibrational_excitations`.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """
        raise NotImplementedError()
