# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Eigenstate results module."""

from typing import Optional, List, Union
import inspect
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.algorithms import AlgorithmResult
from qiskit.opflow import OperatorBase


class EigenstateResult(AlgorithmResult):
    """The eigenstate result interface."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenenergies: Optional[np.ndarray] = None
        self._eigenstates: Optional[List[Union[str,
                                               dict, Result, list, np.ndarray, Statevector,
                                               QuantumCircuit,
                                               Instruction, OperatorBase]]] = None
        self._aux_operator_eigenvalues: Optional[List[float]] = None
        self._raw_result: Optional[AlgorithmResult] = None

    @property
    def eigenenergies(self) -> Optional[np.ndarray]:
        """ returns eigen energies """
        return self._eigenenergies

    @eigenenergies.setter
    def eigenenergies(self, value: np.ndarray) -> None:
        """ set eigen energies """
        self._eigenenergies = value

    @property
    def eigenstates(self) -> Optional[List[Union[str, dict, Result, list, np.ndarray, Statevector,
                                                 QuantumCircuit, Instruction, OperatorBase]]]:
        """ returns eigen states """
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: List[Union[str, dict, Result, list, np.ndarray, Statevector,
                                            QuantumCircuit, Instruction, OperatorBase]]) -> None:
        """ set eigen states """
        self._eigenstates = value

    @property
    def groundenergy(self) -> Optional[float]:
        """ returns ground energy """
        energies = self.eigenenergies
        if energies:
            return energies[0].real
        return None

    @property
    def groundstate(self) -> Optional[Union[str, dict, Result, list, np.ndarray, Statevector,
                                            QuantumCircuit, Instruction, OperatorBase]]:
        """ returns ground state """
        states = self.eigenstates
        if states:
            return states[0]
        return None

    @property
    def aux_operator_eigenvalues(self) -> Optional[List[float]]:
        """ return aux operator eigen values """
        return self._aux_operator_eigenvalues

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: List[float]) -> None:
        """ set aux operator eigen values """
        self._aux_operator_eigenvalues = value

    @property
    def raw_result(self) -> Optional[AlgorithmResult]:
        """Returns the raw algorithm result."""
        return self._raw_result

    @raw_result.setter
    def raw_result(self, result: AlgorithmResult) -> None:
        self._raw_result = result

    def combine(self, result: AlgorithmResult) -> None:
        """
        Any property from the argument that exists in the receiver is
        updated.
        Args:
            result: Argument result with properties to be set.
        Raises:
            TypeError: Argument is None
        """
        if result is None:
            raise TypeError('Argument result expected.')
        if result == self:
            return

        # find any result public property that exists in the receiver
        for name, value in inspect.getmembers(result):
            if not name.startswith('_') and \
                    not inspect.ismethod(value) and not inspect.isfunction(value) and \
                    hasattr(self, name):
                try:
                    setattr(self, name, value)
                except AttributeError:
                    # some attributes may be read only
                    pass
