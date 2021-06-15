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

""" The Unitary Vibrational Coupled-Cluster Single and Double excitations Ansatz. """

from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import logging

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliTrotterEvolution
from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import SecondQuantizedOp, VibrationalOp
from qiskit_nature.converters.second_quantization import QubitConverter
from .evolved_operator_ansatz import EvolvedOperatorAnsatz
from .utils.vibration_excitation_generator import generate_vibration_excitations

logger = logging.getLogger(__name__)


class UVCC(EvolvedOperatorAnsatz):
    """
    This trial wavefunction is a Unitary Vibrational Coupled-Cluster Single and Double excitations
    ansatz.
    For more information, see Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    EXCITATION_TYPE = {
        "s": 1,
        "d": 2,
        "t": 3,
        "q": 4,
    }

    def __init__(
        self,
        qubit_converter: Optional[QubitConverter] = None,
        num_modals: Optional[List[int]] = None,
        excitations: Optional[
            Union[
                str,
                int,
                List[int],
                Callable[
                    [int, Tuple[int, int]],
                    List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                ],
            ]
        ] = None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
    ):
        """

        Args:
            qubit_converter: the QubitConverter instance which takes care of mapping a
                :class:`~.SecondQuantizedOp` to a :class:`PauliSumOp` as well as performing all
                configured symmetry reductions on it.
            num_modals: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode num_modals = [4,4,4]
            excitations: this can be any of the following types:

                :`str`: which contains the types of excitations. Allowed characters are
                    + `s` for singles
                    + `d` for doubles
                    + `t` for triples
                    + `q` for quadruples
                :`int`: a single, positive integer which denotes the number of excitations
                    (1 == `s`, etc.)
                :`List[int]`: a list of positive integers generalizing the above
                :`Callable`: a function which is used to generate the excitations.
                    The callable must take the __keyword__ argument `num_modals` `num_particles`
                    (with identical types to those explained above) and must return a
                    `List[Tuple[Tuple[int, ...], Tuple[int, ...]]]`. For more information on how to
                    write such a callable refer to the default method
                    :meth:`generate_vibration_excitations`.
            reps: number of repetitions of basic module
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
        """
        self._qubit_converter = qubit_converter
        self._num_modals = num_modals
        self._excitations = excitations

        super().__init__(reps=reps, evolution=PauliTrotterEvolution(), initial_state=initial_state)

        # We cache these, because the generation may be quite expensive (depending on the generator)
        # and the user may want quick access to inspect these. Also, it speeds up testing for the
        # same reason!
        self._excitation_ops: List[SecondQuantizedOp] = None

    @property
    def qubit_converter(self) -> Optional[QubitConverter]:
        """The qubit operator converter."""
        return self._qubit_converter

    @qubit_converter.setter
    def qubit_converter(self, conv: QubitConverter) -> None:
        """Sets the qubit operator converter."""
        self._invalidate()
        self._qubit_converter = conv

    @property
    def num_modals(self) -> Optional[List[int]]:
        """The number of modals."""
        return self._num_modals

    @num_modals.setter
    def num_modals(self, num_modals: List[int]) -> None:
        """Sets the number of modals."""
        self._invalidate()
        self._num_modals = num_modals

    @property
    def excitations(self) -> Optional[Union[str, int, List[int], Callable]]:
        """The excitations."""
        return self._excitations

    @excitations.setter
    def excitations(self, exc: Union[str, int, List[int], Callable]) -> None:
        """Sets the excitations."""
        self._invalidate()
        self._excitations = exc

    def _invalidate(self):
        self._excitation_ops = None
        super()._invalidate()

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        if self.num_modals is None or any(b < 0 for b in self.num_modals):
            if raise_on_failure:
                raise ValueError(
                    "The number of modals cannot contain negative values but is ",
                    self.num_modals,
                )
            return False

        if self.excitations is None:
            if raise_on_failure:
                raise ValueError("The excitations cannot be `None`.")
            return False

        if self.qubit_converter is None:
            if raise_on_failure:
                raise ValueError("The qubit_converter cannot be `None`.")
            return False

        return True

    def _build(self) -> None:
        if self._data is not None:
            return

        if self.operators is None or self.operators == [None]:
            excitation_ops = self.excitation_ops()

            logger.debug("Converting SecondQuantizedOps into PauliSumOps...")
            self.operators = self.qubit_converter.convert_match(excitation_ops, suppress_none=True)

        logger.debug("Building QuantumCircuit...")
        super()._build()

    def excitation_ops(self) -> List[SecondQuantizedOp]:
        """Parses the excitations and generates the list of operators.

        Raises:
            QiskitNatureError: if invalid excitations are specified.

        Returns:
            The list of generated excitation operators.
        """
        if self._excitation_ops is not None:
            return self._excitation_ops

        excitations = self._get_excitation_list()

        logger.debug("Converting excitations into SecondQuantizedOps...")
        excitation_ops = self._build_vibration_excitation_ops(excitations)

        self._excitation_ops = excitation_ops
        return excitation_ops

    def _get_excitation_list(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        generators = self._get_excitation_generators()

        logger.debug("Generating excitation list...")
        excitations = []
        for gen in generators:
            excitations.extend(
                gen(
                    num_modals=self.num_modals,
                )
            )

        return excitations

    def _get_excitation_generators(self) -> List[Callable]:
        logger.debug("Gathering excitation generators...")
        generators: List[Callable] = []

        if isinstance(self.excitations, str):
            for exc in self.excitations:
                generators.append(
                    partial(
                        generate_vibration_excitations,
                        num_excitations=self.EXCITATION_TYPE[exc],
                    )
                )
        elif isinstance(self.excitations, int):
            generators.append(
                partial(
                    generate_vibration_excitations,
                    num_excitations=self.excitations,
                )
            )
        elif isinstance(self.excitations, list):
            for exc in self.excitations:  # type: ignore
                generators.append(
                    partial(
                        generate_vibration_excitations,
                        num_excitations=exc,
                    )
                )
        elif callable(self.excitations):
            generators = [self.excitations]
        else:
            raise QiskitNatureError("Invalid excitation configuration: {}".format(self.excitations))

        return generators

    def _build_vibration_excitation_ops(self, excitations: Sequence) -> List[VibrationalOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        Args:
            excitations: the list of excitations.

        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        operators = []

        sum_modes = sum(self.num_modals)

        for exc in excitations:
            label = ["I"] * sum_modes
            for occ in exc[0]:
                label[occ] = "+"
            for unocc in exc[1]:
                label[unocc] = "-"
            op = VibrationalOp("".join(label), len(self.num_modals), self.num_modals)
            op -= op.adjoint()
            # we need to account for an additional imaginary phase in the exponent (see also
            # `PauliTrotterEvolution.convert`)
            op *= 1j  # type: ignore
            operators.append(op)

        return operators
