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
"""
The Unitary Coupled-Cluster Ansatz.
"""

from __future__ import annotations

import logging
from functools import partial
from itertools import chain
from typing import Callable, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz

from qiskit_nature import QiskitNatureError
from qiskit_nature.deprecation import deprecate_arguments, deprecate_property, warn_deprecated_type
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper, TaperedQubitMapper
from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp

from .utils.fermionic_excitation_generator import generate_fermionic_excitations

logger = logging.getLogger(__name__)


class UCC(EvolvedOperatorAnsatz):
    r"""The Unitary Coupled-Cluster Ansatz. For more information, see [1].

    This ansatz is an ``EvolvedOperatorAnsatz`` given by :math:`e^{T - T^{\dagger}}` where
    :math:`T` is the *cluster operator*. This cluster operator generally consists of excitation
    operators which are generated by
    :meth:`~qiskit_nature.second_q.circuit.library.ansatzes.utils.generate_fermionic_excitations`.

    This method constructs the requested excitations based on a
    :class:`~qiskit_nature.second_q.circuit.library.HartreeFock` reference state by default. When
    setting up a ``VQE`` algorithm using this ansatz and initial state, it is likely you will also
    want to use a :class:`~qiskit_nature.second_q.algorithms.initial_points.HFInitialPoint` that has
    been configured using the corresponding ansatz parameters. This can be done as follows:

    .. code-block:: python

        qubit_mapper = JordanWignerMapper()
        ucc = UCC(4, (2, 2), 'sd', qubit_mapper)
        hf_initial_point = HFInitialPoint()
        hf_initial_point.ansatz = ucc
        initial_point = hf_initial_point.to_numpy_array()
        vqe = VQE(Estimator(), ucc, SLSQP(), initial_point=initial_point)

    You can also use a custom excitation generator method by passing a callable to ``excitations``.

    A utility class :class:`UCCSD` exists, which is equivalent to:

    .. code-block:: python

        uccsd = UCC(excitations='sd', alpha_spin=True, beta_spin=True, max_spin_excitation=None)

    If you want to use a tailored ansatz, you have multiple options to do so. Below, we provide some
    examples:

    .. code-block:: python

        # pure single excitations (equivalent options):
        uccs = UCC(excitations='s')
        uccs = UCC(excitations=1)
        uccs = UCC(excitations=[1])

        # pure double excitations (equivalent options):
        uccd = UCC(excitations='d')
        uccd = UCC(excitations=2)
        uccd = UCC(excitations=[2])

        # combinations of excitations:
        custom_ucc_sd = UCC(excitations='sd')  # see also the convenience sub-class UCCSD
        custom_ucc_sd = UCC(excitations=[1, 2])  # see also the convenience sub-class UCCSD
        custom_ucc_sdt = UCC(excitations='sdt')
        custom_ucc_sdt = UCC(excitations=[1, 2, 3])
        custom_ucc_st = UCC(excitations='st')
        custom_ucc_st = UCC(excitations=[1, 3])

        # you can even define a fully custom list of excitations:

        def custom_excitation_list(num_spatial_orbitals: int,
                                   num_particles: tuple[int, int]
                                   ) -> list[tuple[tuple[Any, ...], ...]]:
            # generate your list of excitations...
            my_excitation_list = [...]
            # For more information about the required format of the return statement, please take a
            # look at the documentation of
            # `qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator`
            return my_excitation_list

        my_custom_ucc = UCC(excitations=custom_excitation_list)

    Keep in mind, that in all of the examples above we have not set any of the following keyword
    arguments, which must be specified before the ansatz becomes usable:

    - ``qubit_mapper``
    - ``num_particles``
    - ``num_spatial_orbitals``

    If you are using this ansatz with a Qiskit Nature algorithm, these arguments will be set for
    you, depending on the rest of the stack.


    References:
        [1] `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_

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
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
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
        alpha_spin: bool = True,
        beta_spin: bool = True,
        max_spin_excitation: int | None = None,
        generalized: bool = False,
        preserve_spin: bool = True,
        include_imaginary: bool = False,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """

        Args:
            num_spatial_orbitals: The number of spatial orbitals.
            num_particles: The tuple of the number of alpha- and beta-spin particles.
            excitations: This can be any of the following types:

                :`str`: Contains the types of excitations. Allowed characters are: ``'s'`` for
                    singles, ``'d'`` for doubles, ``'t'`` for triples, and ``'q'`` for quadruples.
                :`int`: A single, positive integer which denotes the number of excitations
                    (``1 == 's'``, ``2 == 'd'``, etc.)
                :`list[int]`: A list of positive integers generalizing the above to multiple numbers
                    of excitations (``[1, 2] == 'sd'``, etc.)
                :`Callable`: A function which is used to generate the excitations.
                    The callable must take the *keyword* arguments ``num_spatial_orbitals`` and
                    ``num_particles`` (with identical types to those explained above) and must return
                    a ``list[tuple[tuple[int, ...], tuple[int, ...]]]``. For more information on how
                    to write such a callable refer to the default method
                    :meth:`~qiskit_nature.second_q.circuit.library.ansatzes.utils.\
                    generate_fermionic_excitations`.
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` or
                :class:`~qiskit_nature.second_q.mappers.QubitConverter` instance (use of the latter
                is deprecated) which takes care of mapping to a qubit operator.
            alpha_spin: Boolean flag whether to include alpha-spin excitations.
            beta_spin: Boolean flag whether to include beta-spin excitations.
            max_spin_excitation: The largest number of excitations within a spin. E.g. you can set
                this to 1 and ``num_excitations`` to 2 in order to obtain only mixed-spin double
                excitations (alpha,beta) but no pure-spin double excitations (alpha,alpha or
                beta,beta).
            generalized: Boolean flag whether or not to use generalized excitations, which ignore
                the occupation of the spin orbitals. As such, the set of generalized excitations is
                only determined from the number of spin orbitals and independent from the number of
                particles.
            preserve_spin: Boolean flag whether or not to preserve the particle spins.
            include_imaginary: Boolean flag which when set to ``True`` expands the ansatz to include
                imaginary parts using twice the number of free parameters.
            reps: The number of times to repeat the evolved operators.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit. Note that this
                setting does *not* influence the ``excitations``. When relying on the default
                generation method (i.e. not providing a ``Callable`` to ``excitations``), these will
                always be constructed with respect to a
                :class:`~qiskit_nature.second_q.circuit.library.HartreeFock` reference state.
                When setting up a ``VQE`` algorithm using this ansatz and initial state, it is
                likely you will also want to use a
                :class:`~qiskit_nature.second_q.algorithms.initial_points.HFInitialPoint` that has
                been configured using the corresponding ansatz parameters.
            qubit_converter: DEPRECATED The :class:`~qiskit_nature.second_q.mappers.QubitConverter`
                or :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance which takes care of
                mapping to a qubit operator.
        """
        self._qubit_mapper = qubit_mapper
        self._num_particles = num_particles
        self._num_spatial_orbitals = num_spatial_orbitals
        self._excitations = excitations
        self._alpha_spin = alpha_spin
        self._beta_spin = beta_spin
        self._max_spin_excitation = max_spin_excitation
        self._generalized = generalized
        self._preserve_spin = preserve_spin
        self._include_imaginary = include_imaginary

        super().__init__(reps=reps, initial_state=initial_state)

        # To give read access to the actual excitation list that UCC is using.
        self._excitation_list: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None

        # We cache these, because the generation may be quite expensive (depending on the generator)
        # and the user may want quick access to inspect these. Also, it speeds up testing for the
        # same reason!
        self._excitation_ops: list[SparseLabelOp] = None

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
    def qubit_converter(self, conv: QubitConverter | QubitMapper | None) -> None:
        """Sets the qubit operator converter."""
        self.qubit_mapper = conv

    @property
    def qubit_mapper(self) -> QubitConverter | QubitMapper | None:
        """The qubit operator mapper."""
        return self._qubit_mapper

    @qubit_mapper.setter
    def qubit_mapper(self, mapper: QubitConverter | QubitMapper | None) -> None:
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
    def num_spatial_orbitals(self) -> int:
        """The number of spatial orbitals."""
        return self._num_spatial_orbitals

    @num_spatial_orbitals.setter
    def num_spatial_orbitals(self, n: int) -> None:
        """Sets the number of spatial orbitals."""
        self._operators = None
        self._invalidate()
        self._num_spatial_orbitals = n

    @property
    def num_particles(self) -> tuple[int, int]:
        """The number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, n: tuple[int, int]) -> None:
        """Sets the number of particles."""
        self._operators = None
        self._invalidate()
        self._num_particles = n

    @property
    def excitations(self) -> str | int | list[int] | Callable | None:
        """The excitations."""
        return self._excitations

    @excitations.setter
    def excitations(self, exc: str | int | list[int] | Callable | None) -> None:
        """Sets the excitations."""
        self._operators = None
        self._invalidate()
        self._excitations = exc

    @property
    def excitation_list(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]] | None:
        """The excitation list that UCC is using.

        Raises:
            QiskitNatureError: If private the excitation list is ``None``.

        """
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
        operators = super(UCC, self.__class__).operators.__get__(self)

        if operators is None or operators == [None]:
            # If the operators are None build them out if the ucc config checks out ok, otherwise
            # they will be left as None to be built at some later time.
            if self._check_ucc_configuration(raise_on_failure=False):
                # The qubit operators are cached by `EvolvedOperatorAnsatz` class. We only generate
                # them from the `SparseLabelOp`s produced by the generators, if they're not
                # already present. This behavior also enables the adaptive usage of the `UCC` class
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

                if self._include_imaginary:
                    # duplicate each excitation to account for the real and imaginary parts.
                    self._excitation_list = list(
                        chain(*zip(self._excitation_list, self._excitation_list))
                    )

                self._filter_operators(operators=operators)

        return super(UCC, self.__class__).operators.__get__(self)

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
        if not self._check_ucc_configuration(raise_on_failure):
            return False

        return super()._check_configuration(raise_on_failure)

    # pylint: disable=too-many-return-statements
    def _check_ucc_configuration(self, raise_on_failure: bool = True) -> bool:
        # Check the local config, separated out that it can be checked via build
        # or ahead of building operators to make sure everything needed is present.
        if self.num_spatial_orbitals is None:
            if raise_on_failure:
                raise ValueError("The number of spatial orbitals cannot be 'None'.")
            return False

        if self.num_spatial_orbitals <= 0:
            if raise_on_failure:
                raise ValueError(
                    f"The number of spatial orbitals must be > 0 was {self.num_spatial_orbitals}."
                )
            return False

        if self.num_particles is None:
            if raise_on_failure:
                raise ValueError("The number of particles cannot be 'None'.")
            return False

        if any(n < 0 for n in self.num_particles):
            if raise_on_failure:
                raise ValueError(
                    f"The number of particles cannot be smaller than 0 was {self.num_particles}."
                )
            return False

        if any(n >= self.num_spatial_orbitals for n in self.num_particles):
            if raise_on_failure:
                raise ValueError(
                    f"The number of spatial orbitals {self.num_spatial_orbitals}"
                    f"must be greater than number of particles of any spin kind "
                    f"{self.num_particles}."
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

        self._check_excitation_list(excitation_list)

        logger.debug("Converting excitations into SparseLabelOps...")
        excitation_ops = self._build_fermionic_excitation_ops(excitation_list)

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
                    num_spatial_orbitals=self.num_spatial_orbitals,
                    num_particles=self.num_particles,
                )
            )

        return excitations

    def _get_excitation_generators(self) -> list[Callable]:
        logger.debug("Gathering excitation generators...")
        generators: list[Callable] = []

        extra_kwargs = {
            "alpha_spin": self._alpha_spin,
            "beta_spin": self._beta_spin,
            "max_spin_excitation": self._max_spin_excitation,
            "generalized": self._generalized,
            "preserve_spin": self._preserve_spin,
        }

        if isinstance(self.excitations, str):
            for exc in self.excitations:
                generators.append(
                    partial(
                        generate_fermionic_excitations,
                        num_excitations=self._EXCITATION_TYPE[exc],
                        **extra_kwargs,
                    )
                )
        elif isinstance(self.excitations, int):
            generators.append(
                partial(
                    generate_fermionic_excitations, num_excitations=self.excitations, **extra_kwargs
                )
            )
        elif isinstance(self.excitations, list):
            for exc in self.excitations:  # type: ignore
                generators.append(
                    partial(generate_fermionic_excitations, num_excitations=exc, **extra_kwargs)
                )
        elif callable(self.excitations):
            generators = [self.excitations]
        else:
            raise QiskitNatureError(f"Invalid excitation configuration: {self.excitations}")

        return generators

    def _check_excitation_list(self, excitations: Sequence) -> None:
        """Checks the format of the given excitation operators.

        The following conditions are checked:
        - the list of excitations consists of pairs of tuples
        - each pair of excitation indices has the same length
        - the indices within each excitation pair are unique

        Args:
            excitations: the list of excitations

        Raises:
            QiskitNatureError: if format of excitations is invalid
        """
        logger.debug("Checking excitation list...")

        error_message = "{error} in the following UCC excitation: {excitation}"

        for excitation in excitations:
            if len(excitation) != 2:
                raise QiskitNatureError(
                    error_message.format(error="Invalid number of tuples", excitation=excitation)
                    + "; Two tuples are expected, e.g. ((0, 1, 4), (2, 3, 6))"
                )

            if len(excitation[0]) != len(excitation[1]):
                raise QiskitNatureError(
                    error_message.format(
                        error="Different number of occupied and virtual indices",
                        excitation=excitation,
                    )
                )

            if any(i in excitation[0] for i in excitation[1]) or any(
                len(set(indices)) != len(indices) for indices in excitation
            ):
                raise QiskitNatureError(
                    error_message.format(error="Duplicated indices", excitation=excitation)
                )

    def _build_fermionic_excitation_ops(self, excitations: Sequence) -> list[FermionicOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        Args:
            excitations: the list of excitations.

        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        num_spin_orbitals = 2 * self.num_spatial_orbitals
        operators = []

        for exc in excitations:
            label = []
            for occ in exc[0]:
                label.append(f"+_{occ}")
            for unocc in exc[1]:
                label.append(f"-_{unocc}")
            op = FermionicOp({" ".join(label): 1}, num_spin_orbitals=num_spin_orbitals)
            op_adj = op.adjoint()
            # we need to account for an additional imaginary phase in the exponent accumulated from
            # the first-order trotterization routine implemented in Qiskit
            op_minus = 1j * (op - op_adj)
            operators.append(op_minus)

            if self._include_imaginary:
                op_plus = -1 * (op + op_adj)
                operators.append(op_plus)

        return operators
