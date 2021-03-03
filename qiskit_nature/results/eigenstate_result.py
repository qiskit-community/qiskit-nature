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

from typing import Optional, List, Union, Dict, Tuple
import collections
import inspect
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.algorithms import AlgorithmResult
from qiskit.opflow import OperatorBase


class EigenstateResult(AlgorithmResult, collections.UserDict):
    """The eigenstate result interface."""

    def __init__(self, a_dict: Optional[Dict] = None) -> None:
        super().__init__()
        if a_dict:
            self.data.update(a_dict)

    def __setitem__(self, key: object, item: object) -> None:
        raise TypeError("'__setitem__' invalid for this object.")

    def __delitem__(self, key: object) -> None:
        raise TypeError("'__delitem__' invalid for this object.")

    def clear(self) -> None:
        raise TypeError("'clear' invalid for this object.")

    def pop(self, key: object, default: Optional[object] = None) -> object:
        raise TypeError("'pop' invalid for this object.")

    def popitem(self) -> Tuple[object, object]:
        raise TypeError("'popitem' invalid for this object.")

    def update(self, *args, **kwargs) -> None:  # pylint: disable=arguments-differ,signature-differs
        raise TypeError("'update' invalid for this object.")

    def combine(self, result: 'AlgorithmResult') -> None:
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
            if not name.startswith('_') and name != 'data' and \
                    not inspect.ismethod(value) and not inspect.isfunction(value) and \
                    hasattr(self, name):
                if value is None:
                    # Just remove from receiver if it exists
                    # since None is the default value in derived classes for non existent name.
                    if name in self.data:
                        del self.data[name]
                else:
                    self.data[name] = value

    def __contains__(self, key: object) -> bool:
        # subclasses have special __getitem__
        try:
            _ = self.__getitem__(key)
            return True
        except KeyError:
            return False

    @property
    def eigenenergies(self) -> Optional[np.ndarray]:
        """ returns eigen energies """
        return self.get('eigenenergies')

    @eigenenergies.setter
    def eigenenergies(self, value: np.ndarray) -> None:
        """ set eigen energies """
        self.data['eigenenergies'] = value

    @property
    def eigenstates(self) -> Optional[List[Union[str, dict, Result, list, np.ndarray, Statevector,
                                                 QuantumCircuit, Instruction, OperatorBase]]]:
        """ returns eigen states """
        return self.get('eigenstates')

    @eigenstates.setter
    def eigenstates(self, value: List[Union[str, dict, Result, list, np.ndarray, Statevector,
                                            QuantumCircuit, Instruction, OperatorBase]]) -> None:
        """ set eigen states """
        self.data['eigenstates'] = value

    @property
    def groundenergy(self) -> Optional[float]:
        """ returns ground energy """
        energies = self.get('eigenenergies')
        if energies:
            return energies[0].real
        return None

    @property
    def groundstate(self) -> Optional[Union[str, dict, Result, list, np.ndarray, Statevector,
                                            QuantumCircuit, Instruction, OperatorBase]]:
        """ returns ground state """
        states = self.get('eigenstates')
        if states:
            return states[0]
        return None

    @property
    def aux_operator_eigenvalues(self) -> Optional[List[float]]:
        """ return aux operator eigen values """
        return self.get('aux_operator_eigenvalues')

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: List[float]) -> None:
        """ set aux operator eigen values """
        self.data['aux_operator_eigenvalues'] = value

    @property
    def raw_result(self) -> Optional[AlgorithmResult]:
        """Returns the raw algorithm result."""
        return self.get('raw_result')

    @raw_result.setter
    def raw_result(self, result: AlgorithmResult) -> None:
        self.data['raw_result'] = result
