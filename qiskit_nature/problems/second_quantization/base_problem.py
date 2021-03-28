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
from typing import List, Optional, Callable, Union

import numpy as np

from qiskit_nature.drivers import BaseDriver, QMolecule, WatsonHamiltonian
from qiskit_nature.transformers import BaseTransformer


class BaseProblem(ABC):
    """Base Problem"""

    # TODO BaseDriver has no run method
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

    def interpret(self, raw_mes_result):
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        raise NotImplementedError()

    def get_default_filter_criterion(self) -> Optional[Callable[[Union[List, np.ndarray], float,
                                                                 Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        raise NotImplementedError()
