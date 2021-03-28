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
from typing import Union, List, Dict, Tuple, Any

import itertools

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.tools import parallel_map
from qiskit.utils import algorithm_globals

from qiskit_nature import QiskitNatureError
from qiskit_nature.circuit.library.ansatzes import UCC
from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.molecular import fermionic_op_builder


def _build_single_hopping_operator(index, num_particles, num_spin_orbitals,
                                   qubit_converter: QubitConverter):
    h_1 = np.zeros((num_spin_orbitals, num_spin_orbitals), dtype=complex)
    h_2 = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals),
                   dtype=complex)
    z2_symmetries = qubit_converter.z2symmetries
    if len(index) == 2:
        i, j = index
        h_1[i, j] = 4.0
    elif len(index) == 4:
        i, j, k, m = index
        h_2[i, j, k, m] = 16.0
    fer_op = fermionic_op_builder.build_ferm_op_from_ints(h_1, h_2)
    qubit_op = qubit_converter.convert_match(fer_op)

    commutativities = []
    if not z2_symmetries.is_empty():
        for symmetry in z2_symmetries.symmetries:
            symmetry_op = PauliSumOp.from_list([(symmetry.to_label(), 1.0)])
            commuting = qubit_op.primitive.table.commutes_with_all(
                symmetry_op.primitive.table)
            anticommuting = qubit_op.primitive.table.anticommutes_with_all(
                symmetry_op.primitive.table)

            if commuting != anticommuting:  # only one of them is True
                if commuting:
                    commutativities.append(True)
                elif anticommuting:
                    commutativities.append(False)
            else:
                raise QiskitNatureError(
                    "Symmetry {} is nor commute neither anti-commute "
                    "to exciting operator.".format(symmetry.to_label()))

    return qubit_op, commutativities


def build_hopping_operators(qmolecule: QMolecule, qubit_converter: QubitConverter,
                            excitations: Union[str, List[List[int]]] = 'sd'
                            ) -> Tuple[Dict[str, PauliSumOp],
                                       Dict[str, List[bool]],
                                       Dict[str, List[Any]]]:
    """Builds the product of raising and lowering operators (basic excitation operators)

    Args:
        excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
            If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
            Otherwise a list of custom excitations can directly be provided.

    Returns:
        A tuple containing the hopping operators, the types of commutativities and the
        excitation indices.
    """

    num_alpha, num_beta = qmolecule.num_alpha, qmolecule.num_beta
    num_molecular_orbitals = qmolecule.num_molecular_orbitals
    num_spin_orbitals = 2 * num_molecular_orbitals

    if isinstance(excitations, str):
        ansatz = UCC(qubit_converter, [num_alpha, num_beta], num_spin_orbitals, excitations)
        excitations_list = [list(itertools.chain.from_iterable(zip(*exc)))
                            for exc in ansatz._get_excitation_list()]
    else:
        excitations_list = excitations

    size = len(excitations_list)

    # # get all to-be-processed index
    # mus, nus = np.triu_indices(size)

    # build all hopping operators
    hopping_operators: Dict[str, PauliSumOp] = {}
    type_of_commutativities: Dict[str, List[bool]] = {}
    excitation_indices = {}
    to_be_executed_list = []
    for idx in range(size):
        to_be_executed_list += [excitations_list[idx], list(reversed(excitations_list[idx]))]
        hopping_operators['E_{}'.format(idx)] = None
        hopping_operators['Edag_{}'.format(idx)] = None
        type_of_commutativities['E_{}'.format(idx)] = None
        type_of_commutativities['Edag_{}'.format(idx)] = None
        excitation_indices['E_{}'.format(idx)] = excitations_list[idx]
        excitation_indices['Edag_{}'.format(idx)] = list(reversed(excitations_list[idx]))

    result = parallel_map(_build_single_hopping_operator,
                          to_be_executed_list,
                          task_args=(num_alpha + num_beta,
                                     num_spin_orbitals,
                                     qubit_converter),
                          num_processes=algorithm_globals.num_processes)

    for key, res in zip(hopping_operators.keys(), result):
        hopping_operators[key] = res[0]
        type_of_commutativities[key] = res[1]

    return hopping_operators, type_of_commutativities, excitation_indices
