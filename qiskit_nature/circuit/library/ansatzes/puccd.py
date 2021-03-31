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
"""
The paired-UCCD variational form.
"""

from typing import List, Optional, Tuple

import itertools
import logging

from qiskit.circuit import QuantumCircuit
from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter

from .ucc import UCC
from .utils.fermionic_excitation_generator import generate_fermionic_excitations

logger = logging.getLogger(__name__)


class PUCCD(UCC):
    """The PUCCD Ansatz.

    The PUCCD Ansatz (by default) only contains double excitations. Furthermore, it enforces all
    excitations to occur in parallel in the alpha and beta species. For more information see also
    [1].

    Note, that this Ansatz can only work for singlet-spin systems. Therefore, the number of alpha
    and beta electrons must be equal.

    This is a convenience subclass of the UCC Ansatz. For more information refer to :class:`UCC`.

    References:

        [1] https://arxiv.org/abs/1911.10864
    """

    def __init__(self, qubit_converter: Optional[QubitConverter] = None,
                 num_particles: Optional[Tuple[int, int]] = None,
                 num_spin_orbitals: Optional[int] = None,
                 reps: int = 1,
                 initial_state: Optional[QuantumCircuit] = None,
                 include_singles: Tuple[bool, bool] = (False, False),
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

        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.
        """
        self._validate_num_particles(num_particles)
        self._include_singles = include_singles
        super().__init__(qubit_converter=qubit_converter,
                         num_particles=num_particles,
                         num_spin_orbitals=num_spin_orbitals,
                         excitations=self.generate_excitations,
                         alpha_spin=True,
                         beta_spin=True,
                         max_spin_excitation=None,
                         reps=reps,
                         initial_state=initial_state)

    @property
    def include_singles(self) -> Tuple[bool, bool]:
        """Whether to include single excitations."""
        return self._include_singles

    @include_singles.setter
    def include_singles(self, include_singles: Tuple[bool, bool]) -> None:
        """Sets whether to include single excitations."""
        self._include_singles = include_singles

    def generate_excitations(self, num_spin_orbitals: int,
                             num_particles: Tuple[int, int]
                             ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Generates the excitations for the PUCCD Ansatz.

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

        excitations = list()
        excitations.extend(generate_fermionic_excitations(1, num_spin_orbitals, num_particles,
                                                          alpha_spin=self.include_singles[0],
                                                          beta_spin=self.include_singles[1]))

        num_electrons = num_particles[0]
        beta_index_shift = num_spin_orbitals // 2

        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_occ = list(range(num_electrons))
        alpha_unocc = list(range(num_electrons, beta_index_shift))
        # the Cartesian product of these lists gives all possible single alpha-spin excitations
        alpha_excitations = list(itertools.product(alpha_occ, alpha_unocc))
        logger.debug('Generated list of single alpha excitations: %s', alpha_excitations)

        for alpha_exc in alpha_excitations:
            # create the beta-spin excitation by shifting into the upper block-spin orbital indices
            beta_exc = (alpha_exc[0] + beta_index_shift, alpha_exc[1] + beta_index_shift)
            # add the excitation tuple
            occ: Tuple[int, ...]
            unocc: Tuple[int, ...]
            occ, unocc = zip(alpha_exc, beta_exc)
            exc_tuple = (occ, unocc)
            excitations.append(exc_tuple)
            logger.debug('Added the excitation: %s', exc_tuple)

        return excitations

    # TODO: when ooVQE gets refactored, it may turn out that this Ansatz can indeed by used for
    # unrestricted spin systems.
    def _validate_num_particles(self, num_particles):
        try:
            assert num_particles[0] == num_particles[1]
        except AssertionError as exc:
            raise QiskitNatureError(
                'The PUCCD Ansatz only works for singlet-spin systems. However, you specified '
                'differing numbers of alpha and beta electrons:', str(num_particles)
            ) from exc
