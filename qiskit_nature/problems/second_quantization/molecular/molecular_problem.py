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
import itertools
from typing import List, Optional, Tuple

import numpy as np

from qiskit_nature import QMolecule
from qiskit_nature.drivers import FermionicDriver
from qiskit_nature.operators import FermionicOp
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.molecular.fermionic_op_factory import create_fermionic_op, \
    create_fermionic_op_from_integrals
from qiskit_nature.transformations.second_quantization import BaseTransformer


class MolecularProblem:
    """Molecular Problem"""

    def __init__(self, fermionic_driver: FermionicDriver,
                 second_quantized_transformations: Optional[List[BaseTransformer]]):
        """

        Args:
            fermionic_driver: A fermionic driver encoding the molecule information.
            second_quantized_transformations: A list of second quantized transformations to be applied to the molecule.
        """
        self.driver = fermionic_driver
        self.transformers = second_quantized_transformations
        self._q_molecule = self.driver.run()
        self._num_modes = self._q_molecule.one_body_integrals.shape[0]

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations provided.

        Returns:
            A list of `SecondQuantizedOp`.
        """

        q_molecule_transformed = self._transform_q_molecule()

        electronic_fermionic_op = create_fermionic_op(q_molecule_transformed)
        total_magnetization_fermionic_op = self._create_total_magnetization_operator()
        total_angular_momentum_fermionic_op = self._create_total_angular_momentum_operator()
        total_particle_number_fermionic_op = self._create_total_particle_number_operator()

        second_quantized_operators_list = [SecondQuantizedOp([electronic_fermionic_op]),
                                           SecondQuantizedOp([total_magnetization_fermionic_op]),
                                           SecondQuantizedOp([total_angular_momentum_fermionic_op]),
                                           SecondQuantizedOp([total_particle_number_fermionic_op])]

        if q_molecule_transformed.has_dipole_integrals():
            x_dipole_operator, y_dipole_operator, z_dipole_operator = self._create_dipole_operators(
                q_molecule_transformed)
            second_quantized_operators_list += [SecondQuantizedOp([x_dipole_operator]),
                                                SecondQuantizedOp([y_dipole_operator]),
                                                SecondQuantizedOp([z_dipole_operator])]

        return second_quantized_operators_list

    def _transform_q_molecule(self) -> QMolecule:
        q_molecule = self._q_molecule
        for transformer in self.transformers:
            q_molecule = transformer.transform(q_molecule)
        return q_molecule

    # TODO likely extract all below to separate classes
    def _create_dipole_operators(self, q_molecule: QMolecule) -> Tuple[FermionicOp, FermionicOp, FermionicOp]:
        x_dipole_operator = self._create_dipole_operator(q_molecule.x_dipole_integrals)
        y_dipole_operator = self._create_dipole_operator(q_molecule.y_dipole_integrals)
        z_dipole_operator = self._create_dipole_operator(q_molecule.z_dipole_integrals)

        return x_dipole_operator, y_dipole_operator, z_dipole_operator

    def _create_dipole_operator(self, dipole_integrals: np.ndarray) -> FermionicOp:
        return create_fermionic_op_from_integrals(dipole_integrals)

    def _create_total_magnetization_operator(self):
        return create_fermionic_op_from_integrals(*self._calculate_total_magnetization_integrals())

    def _create_total_angular_momentum_operator(self):
        return create_fermionic_op_from_integrals(*self._calculate_total_angular_momentum_integrals())

    def _create_total_particle_number_operator(self):
        return create_fermionic_op_from_integrals(*self._calculate_total_particle_number_integrals())

    def _calculate_total_magnetization_integrals(self):
        modes = self._num_modes
        h_1 = np.eye(modes, dtype=complex) * 0.5
        h_1[modes // 2:, modes // 2:] *= -1.0
        h_2 = np.zeros((modes, modes, modes, modes))

        return h_1, h_2

    def _calculate_total_angular_momentum_integrals(self):
        x_h1, x_h2 = self._calculate_s_x_squared_integrals()
        y_h1, y_h2 = self._calculate_s_y_squared_integrals()
        z_h1, z_h2 = self._calculate_s_z_squared_integrals()
        h_1 = x_h1 + y_h1 + z_h1
        h_2 = x_h2 + y_h2 + z_h2

        return h_1, h_2

    def _calculate_total_particle_number_integrals(self):
        modes = self._num_modes
        h_1 = np.eye(modes, dtype=complex)
        h_2 = np.zeros((modes, modes, modes, modes))

        return h_1, h_2

    # TODO eliminate code duplication below
    def _calculate_s_x_squared_integrals(self):

        num_modes = self._num_modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p + num_modes_2, q, q + num_modes_2] += 1.0
                h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h_2[p + num_modes_2, p, q + num_modes_2, q] += 1.0
            else:
                h_2[p, p + num_modes_2, p, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p, p + num_modes_2, p] -= 1.0
                h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def _calculate_s_y_squared_integrals(self):

        num_modes = self._num_modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p + num_modes_2, q, q + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h_2[p + num_modes_2, p, q + num_modes_2, q] -= 1.0
            else:
                h_2[p, p + num_modes_2, p, p + num_modes_2] += 1.0
                h_2[p + num_modes_2, p, p + num_modes_2, p] += 1.0
                h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def _calculate_s_z_squared_integrals(self):

        num_modes = self._num_modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p, q, q] += 1.0
                h_2[p + num_modes_2, p + num_modes_2, q, q] -= 1.0
                h_2[p, p, q + num_modes_2, q + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2,
                    q + num_modes_2, q + num_modes_2] += 1.0
            else:
                h_2[p, p + num_modes_2, p + num_modes_2, p] += 1.0
                h_2[p + num_modes_2, p, p, p + num_modes_2] += 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2
