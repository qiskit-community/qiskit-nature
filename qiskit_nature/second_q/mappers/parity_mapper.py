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

"""The Parity Mapper."""

from __future__ import annotations

import logging

from functools import lru_cache
from typing import Union, List

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries
from qiskit.quantum_info.operators import Pauli, PauliList, SparsePauliOp

from qiskit_nature.second_q.operators import FermionicOp
from .fermionic_mapper import FermionicMapper

logger = logging.getLogger(__name__)


class ParityMapper(FermionicMapper):  # pylint: disable=missing-class-docstring
    def __init__(self, num_particles: tuple[int, int] | None = None):
        """The Parity fermion-to-qubit mapping.

        When using this mapper, :attr:`num_particles` can optionally be used to apply an additional step
        of reduction after the mapping to pauli operators. The two-qubit reduction tapers two qubits
        (middle and last qubit) when the spin orbitals are ordered in two spin sectors (block spin order)
        and selects the symmetry sector corresponding to the provided number of particles.
        """
        super().__init__(allows_two_qubit_reduction=True)
        self._tapering_values: list | None = None
        self.num_particles = num_particles

    @property
    def num_particles(self) -> tuple[int, int] | None:
        """Get number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value: tuple[int, int] | None) -> None:
        """Set number of particles."""
        self._num_particles = value
        self._tapering_values = None
        if self._num_particles is not None:
            num_alpha = self._num_particles[0]
            num_beta = self._num_particles[1]
            par_1 = 1 if (num_alpha + num_beta) % 2 == 0 else -1
            par_2 = 1 if num_alpha % 2 == 0 else -1
            self._tapering_values = [par_2, par_1]

    @classmethod
    @lru_cache(maxsize=32)
    def pauli_table(cls, nmodes: int) -> list[tuple[Pauli, Pauli]]:
        pauli_table = []

        for i in range(nmodes):
            a_z: Union[List[int], np.ndarray] = [0] * (i - 1) + [1] if i > 0 else []
            a_x: Union[List[int], np.ndarray] = [0] * (i - 1) + [0] if i > 0 else []
            b_z: Union[List[int], np.ndarray] = [0] * (i - 1) + [0] if i > 0 else []
            b_x: Union[List[int], np.ndarray] = [0] * (i - 1) + [0] if i > 0 else []
            a_z = np.asarray(a_z + [0] + [0] * (nmodes - i - 1), dtype=bool)
            a_x = np.asarray(a_x + [1] + [1] * (nmodes - i - 1), dtype=bool)
            b_z = np.asarray(b_z + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_x = np.asarray(b_x + [1] + [1] * (nmodes - i - 1), dtype=bool)
            pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))

        return pauli_table

    def _two_qubit_reduce(self, operator: SparsePauliOp) -> SparsePauliOp:
        """
        Applies the two qubit reduction to the operator. This method hard codes the ``Z2Symmetries``
        corresponding to the spin orbitals ordering. The tapering values required to identify the eigen
        sector of the problem are calculated when :attr:`num_particles` is set.

        Args:
            operator: To be tapered operator.
        Returns:
            A new operator whose qubit number is reduced by 2.
        """
        num_qubits = operator.num_qubits
        last_idx = num_qubits - 1
        mid_idx = num_qubits // 2 - 1
        sq_list = [mid_idx, last_idx]

        # build symmetries, sq_paulis:
        symmetries, sq_paulis = [], []
        for idx in sq_list:
            pauli_str = ["I"] * num_qubits

            pauli_str[idx] = "Z"
            z_sym = "".join(pauli_str)[::-1]
            symmetries.append(z_sym)

            pauli_str[idx] = "X"
            sq_pauli = "".join(pauli_str)[::-1]
            sq_paulis.append(sq_pauli)

        symmetries = PauliList(symmetries)
        sq_paulis = PauliList(sq_paulis)
        z2_symmetries = Z2Symmetries(symmetries, sq_paulis, sq_list, self._tapering_values)

        return z2_symmetries.taper(operator)

    def _map_single(self, second_q_op: FermionicOp) -> PauliSumOp:
        mapped_op = ParityMapper.mode_based_mapping(
            second_q_op, second_q_op.register_length
        ).primitive

        reduced_op = mapped_op
        if self.num_particles is not None:
            if mapped_op.num_qubits > 2:
                reduced_op = self._two_qubit_reduce(mapped_op)
            else:
                logger.warning(
                    "The original qubit operator only contains %s qubits! "
                    "Skipping the requested two-qubit reduction!",
                    mapped_op.num_qubits,
                )

        return PauliSumOp(reduced_op)
