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

""" Compact heuristic ansatz for Chemistry """

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import BlueprintCircuit

from qiskit.circuit import ParameterVector, Parameter


class CHC(BlueprintCircuit):
    """ This trial wavefunction is the Compact Heuristic for Chemistry.

    The trial wavefunction is as defined in
    Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855. It aims at approximating
    the UCC Ansatz for a lower CNOT count.

    Note:
        It is not particle number conserving and the accuracy of the approximation decreases
        with the number of excitations.
    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 excitations: Optional[List[Tuple[Tuple[Any, ...], ...]]] = None,
                 reps: int = 1,
                 ladder: bool = False,
                 initial_state: Optional[QuantumCircuit] = None) -> None:
        """

        Args:
            num_qubits: number of qubits
            excitations: The list of excitations encoded as tuples of tuples. Each tuple in the list
                         is a pair of tuples. The first tuple contains the occupied spin orbital
                         indices whereas the second one contains the indices of the unoccupied spin
                         orbitals.
            reps: number of replica of basic module
            ladder: use ladder of CNOTs between to indices in the entangling block
            initial_state: an initial state to prepend to the variational form
        """

        super().__init__()
        self._reps = reps
        self._bounds = None
        self._ladder = ladder
        self._num_qubits = None
        self._excitations = None
        self._initial_state = None
        self._num_parameters = None
        self._ordered_parameters = ParameterVector(name='θ')
        self._support_parameterized_circuit = True

        if num_qubits is not None:
            self.num_qubits = num_qubits

        if excitations is not None:
            self.excitations = excitations

        if initial_state is not None:
            self._initial_state = initial_state

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the variational form.

        Returns:
           int:  An integer indicating the number of qubits.
        """
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits of the variational form.

        Args:
           num_qubits: An integer indicating the number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits
            self.qregs = [QuantumRegister(num_qubits, name='q')]

    @property
    def excitations(self) -> List[Tuple[Tuple[Any, ...], ...]]:
        """The excitation indices to be included in the circuit."""
        return self._excitations

    @excitations.setter
    def excitations(self, excitations: List[Tuple[Tuple[Any, ...], ...]]) -> None:
        """Sets the excitation indices to be included in the circuit."""
        self._invalidate()
        self._excitations = excitations
        self._num_parameters = len(excitations) * self._reps
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters

    @property
    def initial_state(self) -> QuantumCircuit:
        """The initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        """Sets the initial state."""
        self._invalidate()
        self._initial_state = initial_state

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the CHC class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the number of qubits is not specified.
            ValueError: If the number of parameters is not specified.
            ValueError: If the excitation list is not specified.
        """
        error_msg = 'The %s is None but must be set before the circuit can be built.'
        if self._num_qubits is None:
            if raise_on_failure:
                raise ValueError(error_msg, 'number of qubits')
            return False

        if self._num_parameters is None:
            if raise_on_failure:
                raise ValueError(error_msg, 'number of parameters')
            return False

        if self._excitations is None:
            if raise_on_failure:
                raise ValueError(error_msg, 'excitation list')
            return False

        return True

    @property
    def ordered_parameters(self) -> List[Parameter]:
        """The parameters used in the underlying circuit.

        This includes float values and duplicates.

        For more details see :class:`~.NLocal`.

        Returns:
            The parameters objects used in the circuit.
        """
        # TODO: check doc-string
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self._num_parameters)
            return list(self._ordered_parameters)

        return self._ordered_parameters

    @ordered_parameters.setter
    def ordered_parameters(self, parameters: Union[ParameterVector, List[Parameter]]
                           ) -> None:
        """Set the parameters used in the underlying circuit.

        Args:
            The parameters to be used in the underlying circuit.

        Raises:
            ValueError: If the length of ordered parameters does not match the number of
                parameters in the circuit and they are not a ``ParameterVector`` (which could
                be resized to fit the number of parameters).
        """
        if not isinstance(parameters, ParameterVector) \
                and len(parameters) != self._num_parameters:
            raise ValueError('The length of ordered parameters must be equal to the number of '
                             'parameters in the circuit ({}), but is {}'.format(
                                 self._num_parameters, len(parameters)
                             ))
        self._ordered_parameters = parameters
        self._invalidate()

    def _build(self) -> None:
        """
        Construct the variational form, given its parameters.

        Args:
            parameters: circuit parameters
            q: Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: only supports single and double excitations at the moment.
        """
        if self._data is not None:  # type: ignore
            return

        self._check_configuration()
        self._data = []  # type: ignore

        parameters = self.ordered_parameters
        q = self.qubits

        if isinstance(self._initial_state, QuantumCircuit):
            self.append(self._initial_state.to_gate(), range(self._initial_state.num_qubits))

        count = 0
        for _ in range(self._reps):
            for exc in self.excitations:
                occ, unocc = exc
                if len(occ) == 1:

                    i = occ[0]
                    r = unocc[0]

                    self.p(-parameters[count] / 4 + np.pi / 4, q[i])
                    self.p(-parameters[count] / 4 - np.pi / 4, q[r])

                    self.h(q[i])
                    self.h(q[r])

                    if self._ladder:
                        for qubit in range(i, r):
                            self.cx(q[qubit], q[qubit + 1])
                    else:
                        self.cx(q[i], q[r])

                    self.p(parameters[count], q[r])

                    if self._ladder:
                        for qubit in range(r, i, -1):
                            self.cx(q[qubit - 1], q[qubit])
                    else:
                        self.cx(q[i], q[r])

                    self.h(q[i])
                    self.h(q[r])

                    self.p(-parameters[count] / 4 - np.pi / 4, q[i])
                    self.p(-parameters[count] / 4 + np.pi / 4, q[r])

                elif len(occ) == 2:

                    i = occ[0]
                    r = unocc[0]
                    j = occ[1]
                    s = unocc[1]  # pylint: disable=locally-disabled, invalid-name

                    self.sdg(q[r])

                    self.h(q[i])
                    self.h(q[r])
                    self.h(q[j])
                    self.h(q[s])

                    if self._ladder:
                        for qubit in range(i, r):
                            self.cx(q[qubit], q[qubit+1])
                            self.barrier(q[qubit], q[qubit+1])
                    else:
                        self.cx(q[i], q[r])
                    self.cx(q[r], q[j])
                    if self._ladder:
                        for qubit in range(j, s):
                            self.cx(q[qubit], q[qubit+1])
                            self.barrier(q[qubit], q[qubit + 1])
                    else:
                        self.cx(q[j], q[s])

                    self.p(parameters[count], q[s])

                    if self._ladder:
                        for qubit in range(s, j, -1):
                            self.cx(q[qubit-1], q[qubit])
                            self.barrier(q[qubit-1], q[qubit])
                    else:
                        self.cx(q[j], q[s])
                    self.cx(q[r], q[j])
                    if self._ladder:
                        for qubit in range(r, i, -1):
                            self.cx(q[qubit-1], q[qubit])
                            self.barrier(q[qubit - 1], q[qubit])
                    else:
                        self.cx(q[i], q[r])

                    self.h(q[i])
                    self.h(q[r])
                    self.h(q[j])
                    self.h(q[s])

                    self.p(-parameters[count] / 2 + np.pi / 2, q[i])
                    self.p(-parameters[count] / 2 + np.pi, q[r])

                else:
                    raise ValueError('Limited to single and double excitations, '
                                     'higher order is not implemented')

                count += 1
