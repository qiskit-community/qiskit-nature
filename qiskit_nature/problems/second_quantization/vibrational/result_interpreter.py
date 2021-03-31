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
"""The Vibrational Result Interpreter class."""
from typing import Union

import numpy as np
from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult

from qiskit_nature.results import EigenstateResult, VibrationalStructureResult


def _interpret(num_modes: int, raw_result: Union[EigenstateResult, EigensolverResult,
                                                 MinimumEigensolverResult]) -> \
        VibrationalStructureResult:
    """Interprets an EigenstateResult in the context of this transformation.
           Args:
               num_modes: number of modes.
               raw_result: an eigenstate result object.
           Returns:
               An vibronic structure result.
           """

    eigenstate_result = _interpret_raw_result(raw_result)
    result = _interpret_vibr_struct_result(eigenstate_result)
    _interpret_aux_ops_results(num_modes, result)

    return result


def _interpret_raw_result(raw_result):
    eigenstate_result = None
    if isinstance(raw_result, EigenstateResult):
        eigenstate_result = raw_result
    elif isinstance(raw_result, EigensolverResult):
        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_result
        eigenstate_result.eigenenergies = raw_result.eigenvalues
        eigenstate_result.eigenstates = raw_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
    elif isinstance(raw_result, MinimumEigensolverResult):
        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_result
        eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
        eigenstate_result.eigenstates = [raw_result.eigenstate]
        eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
    return eigenstate_result


def _interpret_vibr_struct_result(eigenstate_result):
    result = VibrationalStructureResult()
    result.combine(eigenstate_result)
    result.computed_vibronic_energies = eigenstate_result.eigenenergies
    return result


def _interpret_aux_ops_results(num_modes, result):
    if result.aux_operator_eigenvalues is not None:
        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore

        result.num_occupied_modals_per_mode = []
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            occ_modals = []
            for mode in range(num_modes):
                if aux_op_eigenvalues[mode] is not None:
                    occ_modals.append(aux_op_eigenvalues[mode][0].real)  # type: ignore
                else:
                    occ_modals.append(None)
            result.num_occupied_modals_per_mode.append(occ_modals)  # type: ignore
