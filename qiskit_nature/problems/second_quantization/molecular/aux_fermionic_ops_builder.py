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
from typing import Tuple

from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators import FermionicOp
from qiskit_nature.problems.second_quantization.molecular.fermionic_op_builder import \
    build_ferm_op_from_ints
from qiskit_nature.problems.second_quantization.molecular.integrals_calculators import \
    calc_total_magnetization_ints, calc_total_ang_momentum_ints, calc_total_particle_num_ints


# TODO name this file better

def create_all_aux_operators(q_molecule):
    aux_second_quantized_ops_list = [create_total_particle_number_operator(q_molecule),
                                     create_total_angular_momentum_operator(q_molecule),
                                     create_total_magnetization_operator(q_molecule),
                                     ]

    if q_molecule.has_dipole_integrals():
        x_dipole_operator, y_dipole_operator, z_dipole_operator = create_dipole_operators(
            q_molecule)
        aux_second_quantized_ops_list += [x_dipole_operator,
                                          y_dipole_operator,
                                          z_dipole_operator]
    return aux_second_quantized_ops_list


def create_dipole_operators(q_molecule: QMolecule) -> \
        Tuple[FermionicOp, FermionicOp, FermionicOp]:
    x_dipole_operator = build_ferm_op_from_ints(q_molecule.x_dipole_integrals)
    y_dipole_operator = build_ferm_op_from_ints(q_molecule.y_dipole_integrals)
    z_dipole_operator = build_ferm_op_from_ints(q_molecule.z_dipole_integrals)

    return x_dipole_operator, y_dipole_operator, z_dipole_operator


def create_total_magnetization_operator(q_molecule: QMolecule) -> FermionicOp:
    num_modes = q_molecule.one_body_integrals.shape[0]
    return build_ferm_op_from_ints(*calc_total_magnetization_ints(num_modes))


def create_total_angular_momentum_operator(q_molecule: QMolecule) -> FermionicOp:
    num_modes = q_molecule.one_body_integrals.shape[0]
    return build_ferm_op_from_ints(*calc_total_ang_momentum_ints(num_modes))


def create_total_particle_number_operator(q_molecule: QMolecule) -> FermionicOp:
    num_modes = q_molecule.one_body_integrals.shape[0]
    return build_ferm_op_from_ints(*calc_total_particle_num_ints(num_modes))
