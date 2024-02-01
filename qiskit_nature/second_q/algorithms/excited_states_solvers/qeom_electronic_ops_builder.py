# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility methods to build fermionic hopping operators."""

from __future__ import annotations

from typing import Callable

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import QubitMapper, TaperedQubitMapper
from qiskit_nature.utils import _parallel_map


def build_electronic_ops(
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    excitations: str
    | int
    | list[int]
    | Callable[
        [int, tuple[int, int]],
        list[tuple[tuple[int, ...], tuple[int, ...]]],
    ],
    qubit_mapper: QubitMapper,
) -> tuple[
    dict[str, SparsePauliOp],
    dict[str, list[bool]],
    dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
]:
    # pylint: disable=unused-argument
    """Builds the product of raising and lowering operators (basic excitation operators)

    Args:
        num_spatial_orbitals: The number of spatial orbitals.
        num_particles: The number of alpha- and beta-spin particles as a tuple.
        excitations: The types of excitations to consider. The simple cases for this input are:
            - a `str` containing any of the following characters: `s`, `d`, `t` or `q`.
            - a single, positive `int` denoting the excitation type (1 == `s`, etc.).
            - a list of positive integers.
            - and finally a callable which can be used to specify a custom list of excitations.
              For more details on how to write such a function refer to the default method,
              :meth:`generate_fermionic_excitations`.
        qubit_mapper: The ``QubitMapper`` to use for mapping.

    Returns:
        A tuple containing the hopping operators, the types of commutativities and the excitation
        indices.
    """

    num_alpha, num_beta = num_particles

    ansatz = UCC(num_spatial_orbitals, (num_alpha, num_beta), excitations, qubit_mapper)
    excitations_list = ansatz._get_excitation_list()
    size = len(excitations_list)

    # build all hopping operators
    hopping_operators: dict[str, SparsePauliOp] = {}
    type_of_commutativities: dict[str, list[bool]] = {}
    excitation_indices: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    to_be_executed_list = []
    for idx in range(size):
        to_be_executed_list += [excitations_list[idx], excitations_list[idx][::-1]]
        hopping_operators[f"E_{idx}"] = None
        hopping_operators[f"Edag_{idx}"] = None
        type_of_commutativities[f"E_{idx}"] = None
        type_of_commutativities[f"Edag_{idx}"] = None
        excitation_indices[f"E_{idx}"] = excitations_list[idx]
        excitation_indices[f"Edag_{idx}"] = excitations_list[idx][::-1]

    result = _parallel_map(
        _build_single_hopping_operator,
        to_be_executed_list,
        task_args=(num_spatial_orbitals, qubit_mapper),
    )

    for key, res in zip(hopping_operators.keys(), result):
        hopping_operators[key] = res[0]
        type_of_commutativities[key] = res[1]

    return hopping_operators, type_of_commutativities, excitation_indices


def _build_single_hopping_operator(
    excitation: tuple[tuple[int, ...], tuple[int, ...]],
    num_spatial_orbitals: int,
    qubit_mapper: QubitMapper,
) -> tuple[SparsePauliOp, list[bool]]:
    label = []
    for occ in excitation[0]:
        label.append(f"+_{occ}")
    for unocc in excitation[1]:
        label.append(f"-_{unocc}")
    fer_op = FermionicOp({" ".join(label): 1.0}, num_spin_orbitals=2 * num_spatial_orbitals)

    if isinstance(qubit_mapper, TaperedQubitMapper):
        qubit_op = qubit_mapper.map_clifford(fer_op)
        # Because the clifford conversion was already done, the commutativity information are based
        # on the single qubit pauli objects.
        symmetries_for_commutativity = qubit_mapper.z2symmetries.sq_paulis
    else:
        qubit_op = qubit_mapper.map(fer_op)
        symmetries_for_commutativity = []

    commutativities = []
    if not len(symmetries_for_commutativity) == 0:
        for symmetry in symmetries_for_commutativity:
            symmetry_op = SparsePauliOp.from_list([(symmetry.to_label(), 1.0)])
            paulis = qubit_op.paulis
            len_paulis = len(paulis)
            commuting = len(paulis.commutes_with_all(symmetry_op.paulis)) == len_paulis
            anticommuting = len(paulis.anticommutes_with_all(symmetry_op.paulis)) == len_paulis

            if commuting != anticommuting:  # only one of them is True
                if commuting:
                    commutativities.append(True)
                elif anticommuting:
                    commutativities.append(False)
            else:
                raise QiskitNatureError(
                    f"Symmetry {symmetry.to_label()} neither commutes nor anti-commutes "
                    "with excitation operator."
                )
    return qubit_op, commutativities
