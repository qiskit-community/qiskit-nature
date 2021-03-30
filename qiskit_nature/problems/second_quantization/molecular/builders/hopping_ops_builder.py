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

"""Utility methods to build fermionic hopping operators."""

from typing import cast, Callable, Dict, List, Tuple, Union

from qiskit.opflow import PauliSumOp
from qiskit.tools import parallel_map
from qiskit.utils import algorithm_globals

from qiskit_nature import QiskitNatureError
from qiskit_nature.circuit.library.ansatzes import UCC
from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter


def build_hopping_operators(q_molecule: QMolecule, qubit_converter: QubitConverter,
                            excitations: Union[str, int, List[int],
                                               Callable[[int, Tuple[int, int]],
                                                        List[Tuple[Tuple[int, ...],
                                                                   Tuple[int, ...]]]]] = 'sd',
                            ) -> Tuple[Dict[str, PauliSumOp],
                                       Dict[str, List[bool]],
                                       Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """Builds the product of raising and lowering operators (basic excitation operators)

    Args:
        q_molecule: the `QMolecule` specifying the molecular space in which the excitations lie.
        qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The Z2
                         symmetries stored in this instance are the basis for the commutativity
                         information returned by this method.
        excitations: the types of excitations to consider. The simple cases for this input are:
            - a `str` containing any of the following characters: `s`, `d`, `t` or `q`.
            - a single, positive `int` denoting the excitation type (1 == `s`, etc.).
            - a list of positive integers.
            - and finally a callable which can be used to specify a custom list of excitations.
              For more details on how to write such a function refer to the default method,
              :meth:`generate_fermionic_excitations`.

    Returns:
        A tuple containing the hopping operators, the types of commutativities and the excitation
        indices.
    """

    num_alpha, num_beta = q_molecule.num_alpha, q_molecule.num_beta
    num_molecular_orbitals = q_molecule.num_molecular_orbitals
    num_spin_orbitals = 2 * num_molecular_orbitals

    excitations_list: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    if isinstance(excitations, (str, int)) or \
            (isinstance(excitations, list) and all(isinstance(exc, int) for exc in excitations)):
        excitations = cast(Union[str, int, List[int]], excitations)
        ansatz = UCC(qubit_converter, (num_alpha, num_beta), num_spin_orbitals, excitations)
        excitations_list = ansatz._get_excitation_list()
    else:
        excitations_list = cast(List[Tuple[Tuple[int, ...], Tuple[int, ...]]], excitations)

    size = len(excitations_list)

    # build all hopping operators
    hopping_operators: Dict[str, PauliSumOp] = {}
    type_of_commutativities: Dict[str, List[bool]] = {}
    excitation_indices: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    to_be_executed_list = []
    for idx in range(size):
        to_be_executed_list += [excitations_list[idx], excitations_list[idx][::-1]]
        hopping_operators['E_{}'.format(idx)] = None
        hopping_operators['Edag_{}'.format(idx)] = None
        type_of_commutativities['E_{}'.format(idx)] = None
        type_of_commutativities['Edag_{}'.format(idx)] = None
        excitation_indices['E_{}'.format(idx)] = excitations_list[idx]
        excitation_indices['Edag_{}'.format(idx)] = excitations_list[idx][::-1]

    result = parallel_map(_build_single_hopping_operator,
                          to_be_executed_list,
                          task_args=(num_spin_orbitals,
                                     qubit_converter),
                          num_processes=algorithm_globals.num_processes)

    for key, res in zip(hopping_operators.keys(), result):
        hopping_operators[key] = res[0]
        type_of_commutativities[key] = res[1]

    return hopping_operators, type_of_commutativities, excitation_indices


def _build_single_hopping_operator(excitation: Tuple[Tuple[int, ...], Tuple[int, ...]],
                                   num_spin_orbitals: int,
                                   qubit_converter: QubitConverter
                                   ) -> Tuple[PauliSumOp, List[bool]]:
    label = ['I'] * num_spin_orbitals
    for occ in excitation[0]:
        label[occ] = '+'
    for unocc in excitation[1]:
        label[unocc] = '-'
    fer_op = FermionicOp((''.join(label), 4.0 ** len(excitation[0])))

    qubit_op: PauliSumOp = qubit_converter.convert_match(fer_op)
    z2_symmetries = qubit_converter.z2symmetries

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
