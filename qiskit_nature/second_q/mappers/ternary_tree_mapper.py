# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Ternary Tree Mapper."""

from __future__ import annotations

from functools import lru_cache

from qiskit.quantum_info.operators import SparsePauliOp

from .majorana_mapper import MajoranaMapper
from .mode_based_mapper import ModeBasedMapper, PauliType


class TernaryTreeMapper(MajoranaMapper, ModeBasedMapper):
    """Ternary Tree fermion-to-qubit mapping.

    As described by Jiang, Kalev, Mruczkiewicz, and Neven, Quantum 4, 276 (2020),
    preprint at `arXiv:1910.10746 <https://arxiv.org/abs/1910.10746>`_.

    This is a mapper for :class:`~qiskit_nature.second_q.operators.MajoranaOp`.
    For mapping :class:`~qiskit_nature.second_q.operators.FermionicOp` convert
    to a Majorana operator first:

    .. code-block::

            from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
            from qiskit_nature.second_q.mappers import TernaryTreeMapper
            fermionic_op = FermionicOp(...)
            majorana_op = MajoranaOp.from_fermionic_op(fermionic_op)
            majorana_op = majorana_op.index_order().simplify()
            pauli_op = TernaryTreeMapper().map(majorana_op)

    """

    _pauli_priority: str = "ZXY"

    def __init__(self, pauli_priority: str = "ZXY"):
        """
        Use the Pauli priority argument (one of XYZ, XZY, YXZ, YZX, ZXY, ZYX) to influence which
        Pauli operators appear most frequently in the Pauli strings. The default is 'ZXY', due to
        the fact that the Z gate is usually the most directly supported gate.

        Args:
            pauli_priority (str) : Priority with which Pauli operators are assigned.

        Raises:
            ValueError: if pauli_priority is not one of XYZ, XZY, YXZ, YZX, ZXY, ZYX
        """
        super().__init__()
        self._pauli_priority = pauli_priority.upper()
        if self._pauli_priority not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
            raise ValueError("Pauli priority must be one of XYZ, XZY, YXZ, YZX, ZXY, ZYX")

    def pauli_table(self, register_length: int) -> list[tuple[PauliType, PauliType]]:
        """This method is implemented for ``TernaryTreeMapper`` only for compatibility.
        ``TernaryTreeMapper.map`` only uses the ``sparse_pauli_operators`` method which
        overrides the corresponding method in `ModeBasedMapper`.
        """
        pauli_table = []
        for pauli in self._pauli_table(self._pauli_priority, register_length)[1]:
            # creator/annihilator are constructed as (real +/- imaginary) / 2
            # for Majorana ops (self adjoint) we have imaginary = 0 and need
            # to multiply the operators from _pauli_table by 2 to get the correct pre-factor
            pauli_table.append((2 * pauli[0], SparsePauliOp([""])))
        return pauli_table

    def _pauli_string_length(self, register_length: int) -> int:
        return self._pauli_table(self._pauli_priority, register_length)[0]

    @staticmethod
    @lru_cache(maxsize=32)
    def _pauli_table(
        pauli_priority: str, register_length: int
    ) -> tuple[int, list[tuple[PauliType]]]:
        tree_height = 0
        while 3 ** (tree_height + 1) <= register_length + 1:
            tree_height += 1
        add_nodes = register_length + 1 - 3**tree_height

        pauli_x, pauli_y, pauli_z = tuple(pauli_priority)
        pauli_list: list[tuple[str, list, int]] = [("", [], 1)]
        qubit_index = 0
        for _ in range(tree_height):
            new_pauli_list = []
            for paulis, qubits, coeff in pauli_list:
                new_pauli_list.append((paulis + pauli_x, qubits + [qubit_index], coeff))
                new_pauli_list.append((paulis + pauli_y, qubits + [qubit_index], coeff))
                new_pauli_list.append((paulis + pauli_z, qubits + [qubit_index], coeff))
                qubit_index += 1
            pauli_list = new_pauli_list
        while add_nodes > 0:
            paulis, qubits, coeff = pauli_list.pop(0)
            pauli_list.append((paulis + pauli_x, qubits + [qubit_index], coeff))
            pauli_list.append((paulis + pauli_y, qubits + [qubit_index], coeff))
            add_nodes -= 1
            if add_nodes > 0:
                pauli_list.append((paulis + pauli_z, qubits + [qubit_index], coeff))
                add_nodes -= 1
            qubit_index += 1

        num_qubits = qubit_index
        pauli_list.pop()  # only 2n of the 2n+1 operators are independent
        pauli_ops = [(SparsePauliOp.from_sparse_list([pauli], num_qubits),) for pauli in pauli_list]
        return num_qubits, pauli_ops

    def sparse_pauli_operators(
        self, register_length: int
    ) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        times_creation_annihiliation_op = []

        for paulis in self._pauli_table(self._pauli_priority, register_length)[1]:
            # For Majorana ops (self adjoint) pauli_table is a list of 1-element tuples
            creation_op = SparsePauliOp(paulis[0], coeffs=[1])
            times_creation_annihiliation_op.append(creation_op)

        return (times_creation_annihiliation_op, times_creation_annihiliation_op)
