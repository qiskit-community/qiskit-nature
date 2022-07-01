from typing import List, Optional, Tuple, Sequence

import itertools
import logging

from qiskit.circuit import QuantumCircuit
from qiskit_nature import QiskitNatureError
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit_nature.circuit.library.ansatzes.ucc import UCC
from qiskit_nature.circuit.library.ansatzes.utils.fermionic_excitation_generator import (
    generate_fermionic_excitations,
    get_alpha_excitations,
    get_beta_excitations
)
from qiskit_nature.operators.second_quantization import FermionicOp, SecondQuantizedOp

logger = logging.getLogger(__name__)




class SUCC_full(UCC):
    """The SUCC_full Ansatz.
    The SUCC_full (by default) only contains double excitations. Furthermore, it only considers
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
        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.
        """
        self._validate_num_particles(num_particles)
        self._include_singles = include_singles
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

        num_electrons = num_particles[0]
        beta_index_shift = num_spin_orbitals // 2

        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_excitations = get_alpha_excitations(
            num_electrons, num_spin_orbitals, self._generalized
        )
        beta_excitations = get_beta_excitations(
            num_electrons, num_spin_orbitals, self._generalized
        )
        logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)
        logger.debug("Generated list of single beta excitations: %s", beta_excitations)

        # Find all possible double excitations constructed from the list of single excitations.
        # Note, that we use `product` here, in order to also get those double
        # excitations form an alpha excitation and a beta excitation. We will need those in the
        # following post-processing step.
        pool = itertools.product(alpha_excitations, beta_excitations)

        for exc in pool:
            # find the two excitations (Note: SUCCD only works for double excitations!)
            alpha_exc, second_exc = exc[0], exc[1]
            # shift the second excitation into the beta-spin orbital index range
            beta_exc = (
                second_exc[0],
                second_exc[1],
            )
            # add the excitation tuple
            occ: Tuple[int, ...]
            unocc: Tuple[int, ...]
            occ, unocc = zip(alpha_exc, beta_exc)
            exc_tuple = (occ, unocc)
            excitations.append(exc_tuple)
            logger.debug("Added the excitation: %s", exc_tuple)

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
        ### excitations_dictionary: Dict{Int:List[Tuple[Tuple[int, ...], Tuple[int, ...]]]} = {}
        excitations_dictionary={}
        for exc in excitations:
            exc_level = sum(exc[1])
            if exc_level in excitations_dictionary.keys():
                excitations_dictionary[exc_level].append(exc)
            else:
                excitations_dictionary[exc_level] = [exc]
            
            
            
        for exc_level in excitations_dictionary:
            final_op = 0
            for exc in excitations_dictionary[exc_level]:
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
                final_op += op
            operators.append(final_op)

        return operators