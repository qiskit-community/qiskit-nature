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
"""The Electronic Structure Result Interpreter class."""
from typing import cast, Union

import numpy as np
from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult

from qiskit_nature.drivers import QMolecule
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult, DipoleTuple


def _interpret(molecule_data: QMolecule, molecule_data_transformed: QMolecule,
               raw_result: Union[EigenstateResult, EigensolverResult,
                                 MinimumEigensolverResult]) -> \
        ElectronicStructureResult:
    """Interprets an EigenstateResult in the context of this transformation.

    Args:
        molecule_data: data obtained from running a driver.
        molecule_data_transformed: data obtained from transforming molecule data with a transformer.
        raw_result: an eigenstate result object.

    Returns:
        An electronic structure result.
    """
    eigenstate_result = _interpret_raw_result(raw_result)

    result = _interpret_electr_struct_result(eigenstate_result, molecule_data,
                                             molecule_data_transformed)

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
        eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]
    return eigenstate_result


def _interpret_electr_struct_result(eigenstate_result, molecule_data,
                                    molecule_data_transformed):
    q_molecule = cast(QMolecule, molecule_data)
    q_molecule_transformed = cast(QMolecule, molecule_data_transformed)
    result = ElectronicStructureResult()
    _interpret_eigenstate_results(eigenstate_result, result)
    _interpret_q_molecule_results(q_molecule, result)
    _interpret_transformed_results(q_molecule_transformed, result)
    if result.aux_operator_eigenvalues is not None:
        _interpret_aux_ops_results(q_molecule_transformed, result)
    return result


def _interpret_eigenstate_results(eigenstate_result, result):
    result.combine(eigenstate_result)
    result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])


def _interpret_q_molecule_results(q_molecule, result):
    result.hartree_fock_energy = q_molecule.hf_energy
    result.nuclear_repulsion_energy = q_molecule.nuclear_repulsion_energy
    if q_molecule.nuclear_dipole_moment is not None:
        dipole_tuple = tuple(x for x in q_molecule.nuclear_dipole_moment)
        result.nuclear_dipole_moment = cast(DipoleTuple, dipole_tuple)


def _interpret_transformed_results(q_molecule_transformed, result):
    result.extracted_transformer_energies = q_molecule_transformed.energy_shift


def _interpret_aux_ops_results(q_molecule_transformed, result):
    # the first three values are hardcoded to number of particles, angular momentum
    # and magnetization in this order
    result.num_particles = []
    result.total_angular_momentum = []
    result.magnetization = []
    result.computed_dipole_moment = []
    result.extracted_transformer_dipoles = []
    if not isinstance(result.aux_operator_eigenvalues, list):
        aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
    else:
        aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
    for aux_op_eigenvalues in aux_operator_eigenvalues:
        if aux_op_eigenvalues is None:
            continue

        if aux_op_eigenvalues[0] is not None:
            result.num_particles.append(aux_op_eigenvalues[0][0].real)  # type: ignore

        if aux_op_eigenvalues[1] is not None:
            result.total_angular_momentum.append(aux_op_eigenvalues[1][0].real)  # type: ignore

        if aux_op_eigenvalues[2] is not None:
            result.magnetization.append(aux_op_eigenvalues[2][0].real)  # type: ignore

        if len(aux_op_eigenvalues) >= 6 and q_molecule_transformed.has_dipole_integrals:
            _interpret_dipole_results(aux_op_eigenvalues, q_molecule_transformed, result)


def _interpret_dipole_results(aux_op_eigenvalues, q_molecule_transformed, result):
    # the next three are hardcoded to Dipole moments, if they are set check if the names match
    # extract dipole moment in each axis
    dipole_moment = []
    for moment in aux_op_eigenvalues[3:6]:
        if moment is not None:
            dipole_moment += [moment[0].real]  # type: ignore
        else:
            dipole_moment += [None]

    result.reverse_dipole_sign = q_molecule_transformed.reverse_dipole_sign
    result.computed_dipole_moment.append(cast(DipoleTuple, tuple(dipole_moment)))
    result.extracted_transformer_dipoles.append({
        name: cast(DipoleTuple, (q_molecule_transformed.x_dip_energy_shift[name],
                                 q_molecule_transformed.y_dip_energy_shift[name],
                                 q_molecule_transformed.z_dip_energy_shift[name]))
        for name in q_molecule_transformed.x_dip_energy_shift.keys()
    })
