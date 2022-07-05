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
"""
The SUCCD Ansatz.
"""

from typing import List, Optional, Tuple, Sequence, Dict

import itertools
import logging

from qiskit.circuit import QuantumCircuit
from qiskit_nature import QiskitNatureError
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit_nature.circuit.library.ansatzes.ucc import UCC
from qiskit_nature.circuit.library.ansatzes.utils.fermionic_excitation_generator import (
    generate_fermionic_excitations,
    get_alpha_excitations,
)
from qiskit_nature.operators.second_quantization import FermionicOp

logger = logging.getLogger(__name__)


# TODO: figure out how to implement `succ_full`: a variant of this class, which does include also
# the symmetrically mirrored double excitations, but assigns the same circuit parameter to them.


class SUCCD(UCC):
    """The SUCCD Ansatz.
    The SUCCD Ansatz (by default) only contains double excitations. Furthermore, it only considers
    the set of excitations which is symmetrically invariant with respect to spin-flips of both
    particles. For more information see also [1].
    Note, that this Ansatz can only work for singlet-spin systems. Therefore, the number of alpha
    and beta electrons must be equal.
    This is a convenience subclass of the UCC Ansatz. For more information refer to :class:`UCC`.
    References:
        [1] https://arxiv.org/abs/1911.10864
    """

    def __init__(
        self,
        qubit_converter: Optional[QubitConverter] = None,
        num_particles: Optional[Tuple[int, int]] = None,
        num_spin_orbitals: Optional[int] = None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        include_singles: Tuple[bool, bool] = (False, False),
        generalized: bool = False,
        full: bool = False,
    ):
        """
        Args:
            qubit_converter: the QubitConverter instance which takes care of mapping a
                :class:`~.SecondQuantizedOp` to a :class:`PauliSumOp` as well as performing all
                configured symmetry reductions on it.
            num_particles: the tuple of the number of alpha- and beta-spin particles.
            num_spin_orbitals: the number of spin orbitals.
            reps: The number of times to repeat the evolved operators.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            include_singles: enables the inclusion of single excitations per spin species.
            generalized: boolean flag whether or not to use generalized excitations, which ignore
                the occupation of the spin orbitals. As such, the set of generalized excitations is
                only determined from the number of spin orbitals and independent from the number of
                particles.
            full: boolean flag whether or not to use SUCC_full ansatz, which is a variant of SUCCD
                ansatz that including also the symmetrically mirrored double excitations, but
                assigns the same circuit parameter to them.
        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.
        """
        self._validate_num_particles(num_particles)
        self._include_singles = include_singles
        self._full = full
        super().__init__(
            qubit_converter=qubit_converter,
            num_particles=num_particles,
            num_spin_orbitals=num_spin_orbitals,
            excitations=self.generate_excitations,
            alpha_spin=True,
            beta_spin=True,
            max_spin_excitation=None,
            generalized=generalized,
            reps=reps,
            initial_state=initial_state,
        )

    @property
    def include_singles(self) -> Tuple[bool, bool]:
        """Whether to include single excitations."""
        return self._include_singles

    @include_singles.setter
    def include_singles(self, include_singles: Tuple[bool, bool]) -> None:
        """Sets whether to include single excitations."""
        self._include_singles = include_singles

    @property
    def full(self) -> bool:
        """Whether use SUCC_full"""
        return self._full

    @full.setter
    def full(self, full: bool) -> None:
        """Sets whether to use SUCC_full."""
        self._full = full

    @property
    def excitation_list(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """The excitation list that SUCC is using, in the required format."""
        return self.generate_excitations(
            self.num_spin_orbitals,
            self.num_particles,
        )

    def generate_excitations(
        self, num_spin_orbitals: int, num_particles: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Generates the excitations for the SUCCD Ansatz.
        Args:
            num_spin_orbitals: the number of spin orbitals.
            num_particles: the number of alpha and beta electrons. Note, these must be identical for
            this class.
        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.
        Returns:
            The list of excitations encoded as tuples of tuples. Each tuple in the list is a pair of
            tuples. The first tuple contains the occupied spin orbital indices whereas the second
            one contains the indices of the unoccupied spin orbitals.
        """
        self._validate_num_particles(num_particles)

        excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

        excitations.extend(
            generate_fermionic_excitations(
                1,
                num_spin_orbitals,
                num_particles,
                alpha_spin=self.include_singles[0],
                beta_spin=self.include_singles[1],
            )
        )

        if self._full is False:
            num_electrons = num_particles[0]
            beta_index_shift = num_spin_orbitals // 2

            # generate alpha-spin orbital indices for occupied and unoccupied ones
            alpha_excitations = get_alpha_excitations(
                num_electrons, num_spin_orbitals, self._generalized
            )
            logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)

            # Find all possible double excitations constructed from the list of single excitations.
            # Note, that we use `combinations_with_replacement` here, in order to also get those double
            # excitations which excite from the same occupied level twice. We will need those in the
            # following post-processing step.
            pool = itertools.combinations_with_replacement(alpha_excitations, 2)

            for exc in pool:
                # find the two excitations (Note: SUCCD only works for double excitations!)
                alpha_exc, second_exc = exc[0], exc[1]
                # shift the second excitation into the beta-spin orbital index range
                beta_exc = (
                    second_exc[0] + beta_index_shift,
                    second_exc[1] + beta_index_shift,
                )
                # add the excitation tuple
                occ: Tuple[int, ...]
                unocc: Tuple[int, ...]
                occ, unocc = zip(alpha_exc, beta_exc)
                exc_tuple = (occ, unocc)
                excitations.append(exc_tuple)
                logger.debug("Added the excitation: %s", exc_tuple)

        if self._full:
            excitations.extend(
                generate_fermionic_excitations(
                    2,
                    num_spin_orbitals,
                    num_particles,
                )
            )

        return excitations

    def _validate_num_particles(self, num_particles):
        try:
            assert num_particles[0] == num_particles[1]
        except AssertionError as exc:
            raise QiskitNatureError(
                "The SUCCD Ansatz only works for singlet-spin systems. However, you specified "
                "differing numbers of alpha and beta electrons:",
                str(num_particles),
            ) from exc

    def _build_fermionic_excitation_ops(self, excitations: Sequence) -> List[FermionicOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.
        Args:
            excitations: the list of excitations.
        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        operators = []
        excitations_dictionary: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = {}
        """Reform the excitations list to a dictionary. Each items in the dictionary corresponds to
            a parrameter."""
        for exc in excitations:
            exc_level = sum(exc[1])
            """ sum(exc[1]) sums over the indices of the virtual orbitals in which we excite into.
            Thus, the level of an excitations is indicated by this sum. And the excitaions with the
            same level will be assigned the same parameter."""
            if exc_level in excitations_dictionary:
                excitations_dictionary[exc_level].append(exc)
            else:
                excitations_dictionary[exc_level] = [exc]

        for exc_level, exc_level_items in excitations_dictionary.items():
            ops = []
            for exc in exc_level_items:
                label = ["I"] * self.num_spin_orbitals
                for occ in exc[0]:
                    label[occ] = "+"
                for unocc in exc[1]:
                    label[unocc] = "-"
                op = FermionicOp("".join(label), display_format="dense")
                op -= op.adjoint()
                # we need to account for an additional imaginary phase in the exponent (see also
                # `PauliTrotterEvolution.convert`)
                op *= 1j  # type: ignore
                ops.append(op)
            operators.append(sum(ops))

        return operators
