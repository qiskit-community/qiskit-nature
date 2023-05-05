# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
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

from qiskit_nature import QiskitNatureError
from qiskit_nature.deprecation import deprecate_arguments, deprecate_property, warn_deprecated_type
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper, TaperedQubitMapper
from qiskit_nature.second_q.operators import SparseLabelOp, VibrationalOp

from .utils.vibration_excitation_generator import generate_vibration_excitations

logger = logging.getLogger(__name__)


class UVCC(EvolvedOperatorAnsatz):
    """
    This trial wavefunction is a Unitary Vibrational Coupled-Cluster ansatz.

    This method constructs the requested excitations based on a
    :class:`~qiskit_nature.second_q.circuit.library.VSCF` reference state by default. When setting
    up a ``VQE`` algorithm using this ansatz and initial state, it is likely you will also want to
    use a :class:`~qiskit_nature.second_q.algorithms.initial_points.VSCFInitialPoint` that has been
    configured using the corresponding ansatz parameters. This can be done as follows:

    .. code-block:: python

        qubit_mapper = JordanWignerMapper()
        uvcc = UVCC([2, 2], 'sd', qubit_mapper)
        vscf_initial_point = VSCFInitialPoint()
        vscf_initial_point.ansatz = uvcc
        initial_point = vscf_initial_point.to_numpy_array()
        vqe = VQE(Estimator(), uvcc, SLSQP(), initial_point=initial_point)

    For more information, see Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    _EXCITATION_TYPE = {
        "s": 1,
        "d": 2,
        "t": 3,
        "q": 4,
    }

    @deprecate_arguments(
        "0.6.0",
        {"qubit_converter": "qubit_mapper"},
        additional_msg=(
            ". Additionally, the QubitConverter type in the qubit_mapper argument is deprecated "
            "and support for it will be removed together with the qubit_converter argument."
        ),
    )
    def __init__(
        self,
        num_modals: list[int] | None = None,
        excitations: str
        | int
        | list[int]
        | Callable[
            [int, tuple[int, int]],
            list[tuple[tuple[int, ...], tuple[int, ...]]],
        ]
        | None = None,
        qubit_mapper: QubitConverter | QubitMapper | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """

        Args:
            num_modals: A list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode ``num_modals = [4, 4, 4]``.
            excitations: This can be any of the following types:

                :`str`: Contains the types of excitations. Allowed characters are: ``'s'`` for
                    singles, ``'d'`` for doubles, ``'t'`` for triples, and ``'q'`` for quadruples.
                :`int`: A single, positive integer which denotes the number of excitations
                    (``1 == 's'``, ``2 == 'd'``, etc.).
                :`list[int]`: A list of positive integers generalizing the above to multiple numbers
                    of excitations (``[1, 2] == 'sd'``, etc.).
                :`Callable`: A function which is used to generate the excitations.
                    The callable must take the *keyword* argument ``num_modals``
                    (with identical type to that explained above) and must return a
                    ``list[tuple[tuple[int, ...], tuple[int, ...]]]``. For more information on
                    how to write such a callable refer to the default method :meth:`~qiskit_nature.\
                    second_q.circuit.library.ansatzes.utils.generate_vibration_excitations`.
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` or
                :class:`~qiskit_nature.second_q.mappers.QubitConverter` instance (use of the latter
                is deprecated) which takes care of mapping to a qubit operator.
            reps: The number of repetitions of basic module.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit. Note that this
                setting does *not* influence the ``excitations``. When relying on the default
                generation method (i.e. not providing a ``Callable`` to ``excitations``), these will
                always be constructed with respect to a
                :class:`~qiskit_nature.second_q.circuit.library.VSCF` reference state. When setting
                up a ``VQE`` algorithm using this ansatz and initial state, it is likely you will
                also want to use a
                :class:`~qiskit_nature.second_q.algorithms.initial_points.VSCFInitialPoint` that has
                been configured using the corresponding ansatz parameters.
            qubit_converter: DEPRECATED The :class:`~qiskit_nature.second_q.mappers.QubitConverter`
                or :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance which takes care of
                mapping to a qubit operator.
        """
        self._qubit_mapper = qubit_mapper
        self._num_modals = num_modals
        self._excitations = excitations

        super().__init__(reps=reps, initial_state=initial_state)

        # To give read access to the actual excitation list that UVCC is using.
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None

        # We cache these, because the generation may be quite expensive (depending on the generator)
        # and the user may want quick access to inspect these. Also, it speeds up testing for the
        # same reason!
        self._excitation_ops: list[SparseLabelOp] | None = None

        # Our parent, EvolvedOperatorAnsatz, sets qregs when it knows the
        # number of qubits, which it gets from the operators. Getting the
        # operators here will build them if configuration already allows.
        # This will allow the circuit to be fully built/valid when it's
        # possible at this stage.
        _ = self.operators

    @property
    @deprecate_property("0.6.0", new_name="qubit_mapper")
    def qubit_converter(self) -> QubitConverter | QubitMapper | None:
        """DEPRECATED The qubit operator converter."""
        return self._qubit_mapper

    @qubit_converter.setter
    def qubit_converter(self, conv: QubitConverter | QubitMapper) -> None:
        """Sets the qubit operator converter."""
        self.qubit_mapper = conv

    @property
    def qubit_mapper(self) -> QubitConverter | QubitMapper | None:
        """The qubit operator mapper."""
        return self._qubit_mapper

    @qubit_mapper.setter
    def qubit_mapper(self, mapper: QubitConverter | QubitMapper) -> None:
        """Sets the qubit operator mapper."""
        if isinstance(mapper, QubitConverter):
            warn_deprecated_type(
                "0.6.0",
                argument_name="mapper",
                old_type="QubitConverter",
                new_type="QubitMapper",
            )
        self._operators = None
        self._invalidate()
        self._qubit_mapper = mapper

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
                # them from the `SparseLabelOp`s produced by the generators, if they're not
                # already present. This behavior also enables the adaptive usage of the `UVCC` class
                # by algorithms such as `AdaptVQE`.
                excitation_ops = self.excitation_ops()

                logger.debug("Converting second-quantized into qubit operators...")
                # Convert operators according to saved state in converter from the conversion of the
                # main operator since these need to be compatible. If Z2 Symmetry tapering was done
                # it may be that one or more excitation operators do not commute with the symmetry.
                # The converted operators are maintained at the same index by the converter
                # inserting ``None`` as the result if an operator did not commute. To ensure that
                # the ``excitation_list`` is transformed identically to the operators, we retain
                # ``None`` for non-commuting operators in order to manually remove them in unison.
                if isinstance(self.qubit_mapper, QubitConverter):
                    operators = self.qubit_mapper.convert_match(excitation_ops, suppress_none=False)
                elif isinstance(self.qubit_mapper, TaperedQubitMapper):
                    operators = self.qubit_mapper.map_clifford(excitation_ops)
                    operators = self.qubit_mapper.taper_clifford(operators, suppress_none=False)
                else:
                    operators = self.qubit_mapper.map(excitation_ops)

                self._filter_operators(operators=operators)

        return super(UVCC, self.__class__).operators.__get__(self)

    def _filter_operators(self, operators):
        valid_operators, valid_excitations = [], []
        for op, ex in zip(operators, self._excitation_list):
            if op is not None:
                valid_operators.append(op)
                valid_excitations.append(ex)

        self._excitation_list = valid_excitations
        self.operators = valid_operators

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

        if self.qubit_mapper is None:
            if raise_on_failure:
                raise ValueError("The qubit_mapper cannot be `None`.")
            return False

        return True

    def excitation_ops(self) -> list[SparseLabelOp]:
        """Parses the excitations and generates the list of operators.

        Raises:
            QiskitNatureError: if invalid excitations are specified.

        Returns:
            The list of generated excitation operators.
        """
        if self._excitation_ops is not None:
            return self._excitation_ops

        excitation_list = self._get_excitation_list()

        logger.debug("Converting excitations into SparseLabelOps...")
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
                gen(  # pylint: disable=not-callable
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
                        num_excitations=self._EXCITATION_TYPE[exc],
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

        for exc in excitations:
            label = []
            for occ in exc[0]:
                label.append(f"+_{VibrationalOp.build_dual_index(self.num_modals, occ)}")
            for unocc in exc[1]:
                label.append(f"-_{VibrationalOp.build_dual_index(self.num_modals, unocc)}")
            op = VibrationalOp({" ".join(label): 1}, self.num_modals)
            op -= op.adjoint()
            # we need to account for an additional imaginary phase in the exponent accumulated from
            # the first-order trotterization routine implemented in Qiskit Terra
            op *= 1j  # type: ignore
            operators.append(op)

        return operators
