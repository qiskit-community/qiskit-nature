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

"""The Molecular Problem class."""
from typing import List, Optional, Tuple

import numpy as np

from qiskit_nature import QMolecule
from qiskit_nature.drivers import FermionicDriver
from qiskit_nature.operators import FermionicOp
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.molecular.integrals_calculators \
    .angular_momentum_integrals_calculator import \
    calc_total_ang_momentum_ints
from qiskit_nature.problems.second_quantization.molecular.fermionic_op_builder import \
    build_fermionic_op, \
    build_ferm_op_from_ints
from qiskit_nature.problems.second_quantization.molecular.integrals_calculators \
    .magnetization_integrals_calculator \
    import calc_total_magnetization_ints
from qiskit_nature.problems.second_quantization.molecular.integrals_calculators \
    .particle_number_integrals_calculator \
    import \
    calc_total_particle_num_ints
from qiskit_nature.transformations.second_quantization import BaseTransformer


class MolecularProblem:
    """Molecular Problem"""

    def __init__(self, fermionic_driver: FermionicDriver,
                 second_quantized_transformations: Optional[List[BaseTransformer]]):
        """

        Args:
            fermionic_driver: A fermionic driver encoding the molecule information.
            second_quantized_transformations: A list of second quantized transformations to be
            applied to the molecule.
        """
        self.driver = fermionic_driver
        self.transformers = second_quantized_transformations
        self._q_molecule = self.driver.run()
        self._num_modes = self._q_molecule.one_body_integrals.shape[0]

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp`.
        """

        q_molecule_transformed = self._transform_q_molecule()

        electronic_fermionic_op = build_fermionic_op(q_molecule_transformed)
        total_magnetization_ferm_op = self._create_total_magnetization_operator()
        total_angular_momentum_ferm_op = self._create_total_angular_momentum_operator()
        total_particle_number_ferm_op = self._create_total_particle_number_operator()

        second_quantized_ops_list = [SecondQuantizedOp([electronic_fermionic_op]),
                                     SecondQuantizedOp([total_magnetization_ferm_op]),
                                     SecondQuantizedOp([total_angular_momentum_ferm_op]),
                                     SecondQuantizedOp([total_particle_number_ferm_op])]

        if q_molecule_transformed.has_dipole_integrals():
            x_dipole_operator, y_dipole_operator, z_dipole_operator = self._create_dipole_operators(
                q_molecule_transformed)
            second_quantized_ops_list += [SecondQuantizedOp([x_dipole_operator]),
                                          SecondQuantizedOp([y_dipole_operator]),
                                          SecondQuantizedOp([z_dipole_operator])]

        return second_quantized_ops_list

    def _transform_q_molecule(self) -> QMolecule:
        q_molecule = self._q_molecule
        for transformer in self.transformers:
            q_molecule = transformer.transform(q_molecule)
        return q_molecule

    def _create_dipole_operators(self, q_molecule: QMolecule) -> \
            Tuple[FermionicOp, FermionicOp, FermionicOp]:
        x_dipole_operator = self._create_dipole_operator(q_molecule.x_dipole_integrals)
        y_dipole_operator = self._create_dipole_operator(q_molecule.y_dipole_integrals)
        z_dipole_operator = self._create_dipole_operator(q_molecule.z_dipole_integrals)

        return x_dipole_operator, y_dipole_operator, z_dipole_operator

    def _create_dipole_operator(self, dipole_integrals: np.ndarray) -> FermionicOp:
        return build_ferm_op_from_ints(dipole_integrals)

    def _create_total_magnetization_operator(self):
        return build_ferm_op_from_ints(
            *calc_total_magnetization_ints(self._num_modes))

    def _create_total_angular_momentum_operator(self):
        return build_ferm_op_from_ints(
            *calc_total_ang_momentum_ints(self._num_modes))

    def _create_total_particle_number_operator(self):
        return build_ferm_op_from_ints(
            *calc_total_particle_num_ints(self._num_modes))
