# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hartree-Fock initial state."""

from __future__ import annotations
import warnings
import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit.library import BlueprintCircuit
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils.validation import validate_min

from qiskit_nature.second_q.mappers import (
    BravyiKitaevSuperFastMapper,
    QubitConverter,
    QubitMapper,
    TaperedQubitMapper,
)
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.deprecation import deprecate_arguments, deprecate_property, warn_deprecated_type


class HartreeFock(BlueprintCircuit):
    """A Hartree-Fock initial state."""

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
        qubit_mapper: QubitConverter | QubitMapper | None = None,
        *,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            num_spatial_orbitals: The number of spatial orbitals.
            num_particles: The number of particles as a tuple storing the number of alpha and
                beta-spin electrons in the first and second number, respectively.
            qubit_mapper: A :class:`~qiskit_nature.second_q.mappers.QubitMapper` or a
                :class:`~qiskit_nature.second_q.mappers.QubitConverter` instance (use of the latter
                is deprecated).
            qubit_converter: DEPRECATED A :class:`~qiskit_nature.second_q.mappers.QubitConverter` or
                a :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance.

        Raises:
            NotImplementedError: If ``qubit_mapper`` is (or uses) a
                :class:`~qiskit_nature.second_q.mappers.BravyiKitaevSuperFastMapper`. See
                https://github.com/Qiskit/qiskit-nature/issues/537 for more information.
        """

        super().__init__()
        self._qubit_mapper = qubit_mapper
        self._num_spatial_orbitals = num_spatial_orbitals
        self._num_particles = num_particles
        self._bitstr: list[bool] | None = None

        self._reset_register()

    @property
    @deprecate_property("0.6.0", new_name="qubit_mapper")
    def qubit_converter(self) -> QubitConverter | QubitMapper | None:
        """DEPRECATED The qubit converter."""
        return self._qubit_mapper

    @qubit_converter.setter
    def qubit_converter(self, conv: QubitConverter | QubitMapper | None) -> None:
        """Sets the qubit converter."""
        self.qubit_mapper = conv

    @property
    def qubit_mapper(self) -> QubitConverter | QubitMapper | None:
        """The qubit mapper."""
        return self._qubit_mapper

    @qubit_mapper.setter
    def qubit_mapper(self, mapper: QubitConverter | QubitMapper | None) -> None:
        """Sets the qubit mapper."""
        if isinstance(mapper, QubitConverter):
            warn_deprecated_type(
                "0.6.0",
                argument_name="mapper",
                old_type="QubitConverter",
                new_type="QubitMapper",
            )
        self._invalidate()
        self._qubit_mapper = mapper
        self._reset_register()

    @property
    def num_spatial_orbitals(self) -> int:
        """The number of spatial orbitals."""
        return self._num_spatial_orbitals

    @num_spatial_orbitals.setter
    def num_spatial_orbitals(self, n: int) -> None:
        """Sets the number of spatial orbitals."""
        self._invalidate()
        self._num_spatial_orbitals = n
        self._reset_register()

    @property
    def num_particles(self) -> tuple[int, int]:
        """The number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, n: tuple[int, int]) -> None:
        """Sets the number of particles."""
        self._invalidate()
        self._num_particles = n
        self._reset_register()

    # pylint: disable=too-many-return-statements
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the HartreeFock class is valid.
        Args:
            raise_on_failure: Whether to raise on failure.
        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            returns False. Errors are only raised when raise_on_failure is set to True.

        Raises:
            ValueError: If the number of spatial orbitals is not specified or less than one.
            ValueError: If the number of particles is not specified or less than zero.
            ValueError: If the number of particles of any kind is less than zero.
            ValueError: If the number of spatial orbitals is smaller than the number of particles
                of any kind.
            ValueError: If the qubit mapper is not specified.
            NotImplementedError: If the specified qubit mapper is a
                :class:`~qiskit_nature.second_q.mappers.BravyiKitaevSuperFastMapper` instance.
        """
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

        if any(n > self.num_spatial_orbitals for n in self.num_particles):
            if raise_on_failure:
                raise ValueError(
                    f"The number of spatial orbitals {self.num_spatial_orbitals}"
                    f"must be greater than or equal to the number of particles of "
                    f"any spin kind {self.num_particles}."
                )
            return False

        if self.qubit_mapper is None:
            if raise_on_failure:
                raise ValueError("The qubit mapper cannot be `None`.")
            return False

        if isinstance(self.qubit_mapper, QubitConverter):
            mapper = self.qubit_mapper.mapper
        elif isinstance(self.qubit_mapper, TaperedQubitMapper):
            # we also include the TaperedQubitMapper here, purely for the check done below
            mapper = self.qubit_mapper.mapper
        else:
            mapper = self.qubit_mapper

        if isinstance(mapper, BravyiKitaevSuperFastMapper):
            if raise_on_failure:
                raise NotImplementedError(
                    "Unsupported mapper: ",
                    type(mapper),
                    ". See https://github.com/Qiskit/qiskit-nature/issues/537",
                )
            return False

        return True

    def _reset_register(self):
        """Reset the register and recompute the mapped Hartree-Fock bitstring."""
        self.qregs = []
        self._bitstr = None

        if self._check_configuration(raise_on_failure=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                self._bitstr = hartree_fock_bitstring_mapped(
                    self.num_spatial_orbitals,
                    self.num_particles,
                    self.qubit_mapper,
                    match_convert=True,
                )
            self.qregs = [QuantumRegister(len(self._bitstr), name="q")]

    def _build(self) -> None:
        """
        Construct the Hartree-Fock initial state given its parameters.
        Returns:
            QuantumCircuit: a quantum circuit preparing the Hartree-Fock
            initial state given a number of spatial orbitals, number of particles and
            a qubit mapper.
        """
        if self._is_built:
            return

        super()._build()

        # Construct the circuit for bitstring. Since this is defined as an initial state
        # circuit its assumed that this is applied first to the qubits that are initialized to
        # the zero state. Hence we just need to account for all True entries and set those.
        if self._bitstr is not None:
            for i, bit in enumerate(self._bitstr):
                if bit:
                    self.x(i)


@deprecate_arguments(
    "0.6.0",
    {"qubit_converter": "qubit_mapper"},
    additional_msg=(
        ". Additionally, the QubitConverter type in the qubit_mapper argument is deprecated "
        "and support for it will be removed together with the qubit_converter argument."
    ),
)
@deprecate_arguments(
    "0.6.0",
    {"match_convert": ""},
    additional_msg=(
        ". This argument is no longer necessary when working directly with QubitMapper objects "
        "outside of a QubitConverter because TaperedQubitMapper instances are now built differently"
    ),
)
def hartree_fock_bitstring_mapped(
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    qubit_mapper: QubitConverter | QubitMapper,
    *,
    qubit_converter: QubitConverter | QubitMapper | None = None,
    match_convert: bool = True,
) -> list[bool]:
    # pylint: disable=unused-argument
    """Compute the bitstring representing the mapped Hartree-Fock state for the specified system.

    Args:
        num_spatial_orbitals: The number of spatial orbitals, has a min. value of 1.
        num_particles: The number of particles as a tuple (alpha, beta) containing the number of
            alpha- and  beta-spin electrons, respectively.
        qubit_mapper: A QubitMapper or QubitConverter instance (use of the latter is deprecated).
        qubit_converter: DEPRECATED A QubitConverter or QubitMapper instance.
        match_convert: Whether to use `convert_match` method of the qubit converter (default),
            or just do mapping and possibly two qubit reduction but no tapering. The latter
            is an advanced usage - e.g. if we are trying to auto-select the tapering sector
            then we would not want any match conversion done on a converter that was set to taper.

    Returns:
        The bitstring representing the mapped state of the Hartree-Fock state as array of bools.
    """
    # get the bitstring encoding the Hartree Fock state
    bitstr = hartree_fock_bitstring(num_spatial_orbitals, num_particles)

    # encode the bitstring as a `FermionicOp`
    bitstr_op = FermionicOp(
        {" ".join(f"+_{idx}" for idx, bit in enumerate(bitstr) if bit): 1.0},
        num_spin_orbitals=2 * num_spatial_orbitals,
    )

    # map the `FermionicOp` to a qubit operator
    qubit_op: SparsePauliOp
    if isinstance(qubit_mapper, QubitConverter):
        if match_convert:
            qubit_op = qubit_mapper.convert_match(bitstr_op, check_commutes=False)
        else:
            qubit_op = qubit_mapper.convert_only(bitstr_op, num_particles)
    elif isinstance(qubit_mapper, TaperedQubitMapper):
        # To avoid checking commutativity, we call the two methods separately.
        qubit_op = qubit_mapper.map_clifford(bitstr_op)
        qubit_op = qubit_mapper.taper_clifford(qubit_op, check_commutes=False)
    else:
        qubit_op = qubit_mapper.map(bitstr_op)

    if isinstance(qubit_op, PauliSumOp):
        qubit_op = qubit_op.primitive

    # We check the mapped operator `x` part of the paulis because we want to have particles
    # i.e. True, where the initial state introduced a creation (`+`) operator.
    bits = []
    for bit in qubit_op.paulis.x[0]:
        bits.append(bit)

    return bits


def hartree_fock_bitstring(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> list[bool]:
    """Compute the bitstring representing the Hartree-Fock state for the specified system.

    Args:
        num_spatial_orbitals: The number of spatial orbitals, has a min. value of 1.
        num_particles: The number of particles as a tuple storing the number of alpha- and beta-spin
                       electrons in the first and second number, respectively.

    Returns:
        The bitstring representing the state of the Hartree-Fock state as array of bools.

    Raises:
        ValueError: If the total number of particles is larger than the number of orbitals.
    """
    # validate the input
    validate_min("num_spatial_orbitals", num_spatial_orbitals, 1)
    num_alpha, num_beta = num_particles

    if any(n > num_spatial_orbitals for n in num_particles):
        raise ValueError("# of particles must be less than or equal to # of orbitals.")

    half_orbitals = num_spatial_orbitals
    bitstr = np.zeros(2 * num_spatial_orbitals, bool)
    bitstr[:num_alpha] = True
    bitstr[half_orbitals : (half_orbitals + num_beta)] = True

    return bitstr.tolist()
