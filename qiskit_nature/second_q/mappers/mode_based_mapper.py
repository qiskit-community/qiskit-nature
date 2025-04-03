# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Mode Based Mapper."""

from __future__ import annotations

from typing import Union
from abc import abstractmethod

import numpy as np
from qiskit.quantum_info.operators import Pauli, PauliList, SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.mappers.qubit_mapper import QubitMapper

# Types that can be data for a SparsePauliOp
PauliType = Union[PauliList, SparsePauliOp, Pauli, list, str]


class ModeBasedMapper(QubitMapper):
    """Mapper from ``SparseLabelOp`` to a qubit operator using a Pauli table."""

    def _map_single(
        self, second_q_op: SparseLabelOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        return self.mode_based_mapping(second_q_op, register_length=register_length)

    @abstractmethod
    def pauli_table(self, register_length: int) -> list[tuple[PauliType, PauliType]]:
        r"""Generates a Pauli-lookup table mapping from modes to Pauli operators or pairs of Pauli
        operators.

        This table is a list of tuples :math:`(P, Q)` of two Pauli operators, corresponding to the
        real part :math:`P` and imaginary part :math:`Q` for the respective mode index. These Pauli
        operators are used to construct the creation and annihilation operators
        :math:`(P \pm i Q)/2`.

        The generated table is processed by :meth:`.sparse_pauli_operators`.

        Args:
            register_length: the register length for which to generate the table.

        Returns:
            A list of tuples of two Pauli string operators.
        """

    def sparse_pauli_operators(
        self, register_length: int
    ) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        # pylint: disable=unused-argument
        """Generates the :class:`.SparsePauliOp` terms.

        This uses :meth:`.pauli_table` to construct a list of operators used to
        translate the second-quantization symbols into qubit operators.

        Args:
            register_length: the register length for which to generate the operators.

        Returns:
            Two lists stored in a tuple, consisting of the creation and annihilation  operators,
            applied on the individual modes.
        """
        times_creation_op = []
        times_annihilation_op = []

        for paulis in self.pauli_table(register_length):
            real_part = SparsePauliOp(paulis[0], coeffs=[0.5])
            imag_part = SparsePauliOp(paulis[1], coeffs=[0.5j])

            # The creation operator is given by 0.5*(X - 1j*Y)
            creation_op = real_part - imag_part
            times_creation_op.append(creation_op)

            # The annihilation operator is given by 0.5*(X + 1j*Y)
            annihilation_op = real_part + imag_part
            times_annihilation_op.append(annihilation_op)

        return (times_creation_op, times_annihilation_op)

    def mode_based_mapping(
        self,
        second_q_op: SparseLabelOp,
        register_length: int | None = None,
    ) -> SparsePauliOp:
        # pylint: disable=unused-argument
        """Utility method to map a ``SparseLabelOp`` to a qubit operator using a pauli table.

        Args:
            second_q_op: the `SparseLabelOp` to be mapped.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the operator being mapped. This is possible because the
                ``register_length`` is considered a lower bound.

        Returns:
            The qubit operator corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: If number length of pauli table does not match the number
                of operator modes, or if the operator has unexpected label content
        """
        if register_length is None:
            register_length = second_q_op.register_length

        times_creation_op, times_annihilation_op = self.sparse_pauli_operators(register_length)
        mapped_string_length = times_creation_op[0].num_qubits

        # make sure ret_op_list is not empty by including a zero op
        ret_op_list = [SparsePauliOp("I" * mapped_string_length, coeffs=[0])]

        for terms, coeff in second_q_op.terms():
            # 1. Initialize an operator list with the identity scaled by the `coeff`
            ret_op = SparsePauliOp("I" * mapped_string_length, coeffs=np.array([coeff]))

            # Go through the label and replace the fermion operators by their qubit-equivalent, then
            # save the respective Pauli string in the pauli_str list.
            for term in terms:
                char = term[0]
                position = int(term[1])
                if char in ("+", ""):  # "" for MajoranaOp, creator = annihilator
                    ret_op = ret_op.compose(times_creation_op[position], front=True).simplify()
                elif char == "-":
                    ret_op = ret_op.compose(times_annihilation_op[position], front=True).simplify()
                # catch any disallowed labels
                else:
                    raise QiskitNatureError(
                        f"FermionicOp label included '{char}'. Allowed characters: I, N, E, +, -"
                    )
            ret_op_list.append(ret_op)

        sparse_op = SparsePauliOp.sum(ret_op_list).simplify()
        return sparse_op
