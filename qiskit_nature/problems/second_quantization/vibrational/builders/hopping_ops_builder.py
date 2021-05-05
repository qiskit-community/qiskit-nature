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

"""Utility methods to build vibrational hopping operators."""

from typing import cast, Callable, Dict, List, Tuple, Union

from qiskit.opflow import PauliSumOp
from qiskit.tools import parallel_map
from qiskit.utils import algorithm_globals

from qiskit_nature.circuit.library import UVCC
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.converters.second_quantization import QubitConverter


def _build_qeom_hopping_ops(
    num_modals: List[int],
    qubit_converter: QubitConverter,
    excitations: Union[
        str,
        int,
        List[int],
        Callable[[int, Tuple[int, int]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
    ] = "sd",
) -> Tuple[
    Dict[str, PauliSumOp],
    Dict[str, List[bool]],
    Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
]:
    """
    Args:
        num_modals: the number of modals per mode.
        qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The Z2
                         symmetries stored in this instance are the basis for the commutativity
                         information returned by this method.
        excitations: the types of excitations to consider. The simple cases for this input are:
            - a `str` containing any of the following characters: `s`, `d`, `t` or `q`.
            - a single, positive `int` denoting the excitation type (1 == `s`, etc.).
            - a list of positive integers.
            - and finally a callable which can be used to specify a custom list of excitations.
              For more details on how to write such a function refer to the default method,
              :meth:`generate_vibrational_excitations`.
    Returns:
        Dict of hopping operators, dict of commutativity types and dict of excitation indices
    """

    excitations_list: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    if isinstance(excitations, (str, int)) or (
        isinstance(excitations, list)
        and all(isinstance(exc, int) for exc in excitations)
    ):
        excitations = cast(Union[str, int, List[int]], excitations)
        ansatz = UVCC(qubit_converter, num_modals, excitations)
        excitations_list = ansatz._get_excitation_list()
    else:
        excitations_list = cast(
            List[Tuple[Tuple[int, ...], Tuple[int, ...]]], excitations
        )

    size = len(excitations_list)

    hopping_operators: Dict[str, PauliSumOp] = {}
    excitation_indices: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    to_be_executed_list = []
    for idx in range(size):
        to_be_executed_list += [excitations_list[idx], excitations_list[idx][::-1]]
        hopping_operators["E_{}".format(idx)] = None
        hopping_operators["Edag_{}".format(idx)] = None
        excitation_indices["E_{}".format(idx)] = excitations_list[idx]
        excitation_indices["Edag_{}".format(idx)] = excitations_list[idx][::-1]

    result = parallel_map(
        _build_single_hopping_operator,
        to_be_executed_list,
        task_args=(num_modals, qubit_converter),
        num_processes=algorithm_globals.num_processes,
    )

    for key, res in zip(hopping_operators.keys(), result):
        hopping_operators[key] = res

    # This variable is required for compatibility with the ElectronicStructureProblem
    # at the moment we do not have any type of commutativity in the bosonic case.
    type_of_commutativities: Dict[str, List[bool]] = {}

    return hopping_operators, type_of_commutativities, excitation_indices  # type: ignore


def _build_single_hopping_operator(
    excitation: Tuple[Tuple[int, ...], Tuple[int, ...]],
    num_modals: List[int],
    qubit_converter: QubitConverter,
) -> PauliSumOp:
    sum_modes = sum(num_modals)

    label = ["I"] * sum_modes
    for occ in excitation[0]:
        label[occ] = "+"
    for unocc in excitation[1]:
        label[unocc] = "-"
    vibrational_op = VibrationalOp("".join(label), len(num_modals), num_modals)
    qubit_op: PauliSumOp = qubit_converter.convert_match(vibrational_op)

    return qubit_op
