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

"""Utility methods for the creation of common auxiliary operators."""

from typing import List, Tuple

from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.problems.second_quantization.electronic.builders.fermionic_op_builder import (
    build_ferm_op_from_ints,
)
from qiskit_nature.problems.second_quantization.electronic.integrals_calculators import (
    calc_total_magnetization_ints,
    calc_total_ang_momentum_ints,
    calc_total_particle_num_ints,
)


def _create_all_aux_operators(q_molecule: QMolecule) -> List[FermionicOp]:
    """Generates the common auxiliary operators out of the given QMolecule.

    Args:
        q_molecule: the QMolecule object for which to generate the operators.

    Returns:
        A list of auxiliary FermionicOps. The first three entries will always correspond to the
        particle number, angular momentum and total magnetization operators. If the QMolecule object
        contained dipole integrals, the list will also contain the X, Y and Z dipole operators.
    """
    aux_second_quantized_ops_list = [
        _create_total_particle_num_op(q_molecule),
        _create_total_ang_momentum_op(q_molecule),
        _create_total_magnetization_op(q_molecule),
    ]

    if q_molecule.has_dipole_integrals():
        x_dipole_operator, y_dipole_operator, z_dipole_operator = _create_dipole_ops(q_molecule)
        aux_second_quantized_ops_list += [
            x_dipole_operator,
            y_dipole_operator,
            z_dipole_operator,
        ]
    return aux_second_quantized_ops_list


def _create_dipole_ops(
    q_molecule: QMolecule,
) -> Tuple[FermionicOp, FermionicOp, FermionicOp]:
    x_dipole_operator = build_ferm_op_from_ints(q_molecule.x_dipole_integrals)
    y_dipole_operator = build_ferm_op_from_ints(q_molecule.y_dipole_integrals)
    z_dipole_operator = build_ferm_op_from_ints(q_molecule.z_dipole_integrals)

    return x_dipole_operator, y_dipole_operator, z_dipole_operator


def _create_total_magnetization_op(q_molecule: QMolecule) -> FermionicOp:
    num_modes = q_molecule.one_body_integrals.shape[0]
    return build_ferm_op_from_ints(*calc_total_magnetization_ints(num_modes))


def _create_total_ang_momentum_op(q_molecule: QMolecule) -> FermionicOp:
    num_modes = q_molecule.one_body_integrals.shape[0]
    return build_ferm_op_from_ints(*calc_total_ang_momentum_ints(num_modes))


def _create_total_particle_num_op(q_molecule: QMolecule) -> FermionicOp:
    num_modes = q_molecule.one_body_integrals.shape[0]
    from qiskit_nature.properties.particle_number import ParticleNumber
    return ParticleNumber(num_modes, (1, 1)).second_q_ops()[0]
