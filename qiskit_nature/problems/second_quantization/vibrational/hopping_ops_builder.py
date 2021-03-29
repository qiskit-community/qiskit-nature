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
from typing import List, Union, Tuple, Dict
import itertools

from qiskit.opflow import PauliSumOp
from qiskit.tools import parallel_map
from qiskit.utils import algorithm_globals

from qiskit_nature.circuit.library.ansatzes import UVCC
from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.vibrational.vibrational_op_builder import \
    build_vibrational_op


def _build_single_hopping_operator(watson_hamiltonian: WatsonHamiltonian,
                                   num_modals: Union[int, List[int]],
                                   truncation_order,
                                   qubit_converter: QubitConverter,
                                   basis: List[int] = None):
    """
    Builds a hopping operator given the list of indices (index) that is a single, a double
    or a higher order excitation.
    Args:
        index: the indexes defining the excitation
        basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
            with 4 modals per mode basis = [4,4,4]
        qubit_mapping: the qubits mapping type. Only 'direct' is supported at the moment.
    Returns:
        Qubit operator object corresponding to the hopping operator
    """

    vibrational_op = build_vibrational_op(watson_hamiltonian, num_modals, truncation_order, basis)
    qubit_op = qubit_converter.convert_match(vibrational_op)
    # if len(qubit_op.paulis) == 0: # TODO how to update it?
    #     qubit_op = None

    return qubit_op


def build_hopping_operators(watson_hamiltonian: WatsonHamiltonian,
                            num_modals: Union[int, List[int]],
                            truncation_order,
                            qubit_converter: QubitConverter,
                            excitations: Union[str, List[List[int]]] = 'sd'
                            ) -> Tuple[Dict[str, PauliSumOp], Dict, Dict[str, List[List[int]]]]:
    """
    Args:
        excitations:
    Returns:
        Dict of hopping operators, dict of commutativity types and dict of excitation indices
    """

    if isinstance(excitations, str):
        ansatz = UVCC(qubit_converter, num_modals, excitations)
        excitations_list = [list(itertools.chain.from_iterable(zip(*exc)))
                            for exc in ansatz._get_excitation_list()]
    else:
        excitations_list = excitations

    size = len(excitations_list)

    def _dag_list(extn_lst):
        dag_lst = []
        for lst in extn_lst:
            dag_lst.append([lst[0], lst[2], lst[1]])
        return dag_lst

    hopping_operators: Dict[str, PauliSumOp] = {}
    excitation_indices = {}
    to_be_executed_list = []
    for idx in range(size):
        to_be_executed_list += [excitations_list[idx], _dag_list(excitations_list[idx])]
        hopping_operators['E_{}'.format(idx)] = None
        hopping_operators['Edag_{}'.format(idx)] = None
        excitation_indices['E_{}'.format(idx)] = excitations_list[idx]
        excitation_indices['Edag_{}'.format(idx)] = _dag_list(excitations_list[idx])

    result = parallel_map(_build_single_hopping_operator,
                          to_be_executed_list,
                          task_args=(watson_hamiltonian,
                                     num_modals,
                                     truncation_order,
                                     qubit_converter),
                          num_processes=algorithm_globals.num_processes)

    for key, res in zip(hopping_operators.keys(), result):
        hopping_operators[key] = res

    # This variable is required for compatibility with the FermionicTransformation
    # at the moment we do not have any type of commutativity in the bosonic case.
    type_of_commutativities: Dict[str, List[bool]] = {}

    return hopping_operators, type_of_commutativities, excitation_indices  # type: ignore
