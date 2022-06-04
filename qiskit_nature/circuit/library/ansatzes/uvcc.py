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

""" The Unitary Vibrational Coupled-Cluster Ansatz. """

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.opflow import PauliTrotterEvolution

from qiskit_nature import QiskitNatureError
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.operators.second_quantization import SecondQuantizedOp, VibrationalOp

from .utils.vibration_excitation_generator import generate_vibration_excitations

logger = logging.getLogger(__name__)


class UVCC(EvolvedOperatorAnsatz):
    """
    This trial wavefunction is a Unitary Vibrational Coupled-Cluster ansatz.

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
        qubit_converter: QubitConverter | None = None,
        num_modals: list[int] | None = None,
        excitations: str
        | int
        | list[int]
        | Callable[
            [int, tuple[int, int]],
            list[tuple[tuple[int, ...], tuple[int, ...]]],
        ]
        | None = None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
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
                :`list[int]`: a list of positive integers generalizing the above
                :`Callable`: a function which is used to generate the excitations.
                    The callable must take the __keyword__ argument `num_modals` `num_particles`
                    (with identical types to those explained above) and must return a
                    `list[tuple[tuple[int, ...], tuple[int, ...]]]`. For more information on how to
                    write such a callable refer to the default method
                    :meth:`~qiskit_nature.circuit.library.ansatzes.utils.generate_vibration_excitations`.
            reps: number of repetitions of basic module
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
        """
        self._qubit_converter = qubit_converter
        self._num_modals = num_modals
        self._excitations = excitations

        super().__init__(reps=reps, evolution=PauliTrotterEvolution(), initial_state=initial_state)

        # To give read access to the actual excitation list that UVCC is using.
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None

        # We cache these, because the generation may be quite expensive (depending on the generator)
        # and the user may want quick access to inspect these. Also, it speeds up testing for the
        # same reason!
        self._excitation_ops: list[SecondQuantizedOp] | None = None

        # Our parent, EvolvedOperatorAnsatz, sets qregs when it knows the
        # number of qubits, which it gets from the operators. Getting the
        # operators here will build them if configuration already allows.
        # This will allow the circuit to be fully built/valid when it's
        # possible at this stage.
        _ = self.operators

    @property
    def qubit_converter(self) -> QubitConverter | None:
        """The qubit operator converter."""
        return self._qubit_converter

    @qubit_converter.setter
    def qubit_converter(self, conv: QubitConverter) -> None:
        """Sets the qubit operator converter."""
        self._operators = None
        self._invalidate()
        self._qubit_converter = conv

    @property
    def num_modals(self) -> list[int] | None:
        """The number of modals."""
        return self._num_modals

    @num_modals.setter
    def num_modals(self, num_modals: list[int]) -> None:
        """Sets the number of modals."""
        self._operators = None
        self._invalidate()
        self._num_modals = num_modals

    @property
    def excitations(self) -> str | int | list[int] | Callable | None:
        """The excitations."""
        return self._excitations

    @excitations.setter
    def excitations(self, exc: str | int | list[int] | Callable) -> None:
        """Sets the excitations."""
        self._operators = None
        self._invalidate()
        self._excitations = exc

    @property
    def excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]] | None:
        """The excitation list that UVCC is using."""
        if self._excitation_list is None:
            # If the excitation_list is None build it out alongside the operators if the ucc config
            # checks out ok, otherwise it will be left as None to be built at some later time.
            _ = self.operators
        return self._excitation_list

    @EvolvedOperatorAnsatz.operators.getter
    def operators(self):  # pylint: disable=invalid-overridden-method
        """The operators that are evolved in this circuit.

        Returns:
            list: The operators to be evolved contained in this ansatz or
                  None if the configuration is not complete
        """
        # Overriding the getter to build the operators on demand when they are
        # requested, if they are still set to None.
        operators = super(UVCC, self.__class__).operators.__get__(self)

        if operators is None or operators == [None]:
            # If the operators are None build them out if the uvcc config checks out ok, otherwise
            # they will be left as None to be built at some later time.
            if self._check_uvcc_configuration(raise_on_failure=False):
                # The qubit operators are cached by `EvolvedOperatorAnsatz` class. We only generate
                # them from the `SecondQuantizedOp`s produced by the generators, if they're not
                # already present. This behavior also enables the adaptive usage of the `UVCC` class
                # by algorithms such as `AdaptVQE`.
                excitation_ops = self.excitation_ops()

                logger.debug("Converting SecondQuantizedOps into PauliSumOps...")
                # Convert operators according to saved state in converter from the conversion of the
                # main operator since these need to be compatible. If Z2 Symmetry tapering was done
                # it may be that one or more excitation operators do not commute with the symmetry.
                # The converted operators are maintained at the same index by the converter
                # inserting ``None`` as the result if an operator did not commute. To ensure that
                # the ``excitation_list`` is transformed identically to the operators, we retain
                # ``None`` for non-commuting operators in order to manually remove them in unison.
                operators = self.qubit_converter.convert_match(excitation_ops, suppress_none=False)
                valid_operators, valid_excitations = [], []
                for op, ex in zip(operators, self._excitation_list):
                    if op is not None:
                        valid_operators.append(op)
                        valid_excitations.append(ex)

                self._excitation_list = valid_excitations
                self.operators = valid_operators

        return super(UVCC, self.__class__).operators.__get__(self)

    def _invalidate(self):
        self._excitation_ops = None
        super()._invalidate()

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        # Check our local config is valid first. The super class will check the
        # operators by getting them, and if we detect they are still None they
        # will be built so that its valid check will end up passing in that regard.
        if not self._check_uvcc_configuration(raise_on_failure):
            return False

        return super()._check_configuration(raise_on_failure)

    def _check_uvcc_configuration(self, raise_on_failure: bool = True) -> bool:
        # Check the local config, separated out that it can be checked via build
        # or ahead of building operators to make sure everything needed is present.
        if self.num_modals is None:
            if raise_on_failure:
                raise ValueError("The number of modals cannot be 'None`.")
            return False

        if any(b < 0 for b in self.num_modals):
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

    def excitation_ops(self) -> list[SecondQuantizedOp]:
        """Parses the excitations and generates the list of operators.

        Raises:
            QiskitNatureError: if invalid excitations are specified.

        Returns:
            The list of generated excitation operators.
        """
        if self._excitation_ops is not None:
            return self._excitation_ops

        excitation_list = self._get_excitation_list()

        logger.debug("Converting excitations into SecondQuantizedOps...")
        excitation_ops = self._build_vibration_excitation_ops(excitation_list)

        self._excitation_list = excitation_list
        self._excitation_ops = excitation_ops
        return excitation_ops

    def _get_excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
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

    def _get_excitation_generators(self) -> list[Callable]:
        logger.debug("Gathering excitation generators...")
        generators: list[Callable] = []

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
            raise QiskitNatureError(f"Invalid excitation configuration: {self.excitations}")

        return generators

    def _build_vibration_excitation_ops(self, excitations: Sequence) -> list[VibrationalOp]:
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
