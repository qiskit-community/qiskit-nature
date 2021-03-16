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
from typing import List, Tuple, Optional

from qiskit_nature.drivers.qmolecule import QMolecule
from qiskit_nature.drivers import FermionicDriver
from qiskit_nature.operators import FermionicOp
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.transformers import BaseTransformer
from .integrals_calculators import calc_total_ang_momentum_ints
from .fermionic_op_builder import build_fermionic_op, build_ferm_op_from_ints
from .integrals_calculators import calc_total_magnetization_ints
from .integrals_calculators import calc_total_particle_num_ints


class MolecularProblem:
    """Molecular Problem"""

    def __init__(self, fermionic_driver: FermionicDriver,
                 q_molecule_transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            fermionic_driver: A fermionic driver encoding the molecule information.
            q_molecule_transformers: A list of transformations to be applied to the molecule.
        """
        if q_molecule_transformers is None:
            q_molecule_transformers = []
        self.driver = fermionic_driver
        self.transformers = q_molecule_transformers

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: electronic operator,
            total magnetization operator, total angular momentum operator, total particle number
            operator, and (if available) x, y, z dipole operators.
        """
        q_molecule = self.driver.run()
        q_molecule_transformed = self._transform_q_molecule(q_molecule)
        num_modes = q_molecule_transformed.one_body_integrals.shape[0]

        electronic_fermionic_op = build_fermionic_op(q_molecule_transformed)
        total_magnetization_ferm_op = self._create_total_magnetization_operator(num_modes)
        total_angular_momentum_ferm_op = self._create_total_angular_momentum_operator(num_modes)
        total_particle_number_ferm_op = self._create_total_particle_number_operator(num_modes)

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

    def _transform_q_molecule(self, q_molecule) -> QMolecule:
        #ensure Particle Hole Transformer is executed last
        ParticleHoleTransformer = None

        for transformer in self.transformers:
            if transformer.__class__.__name__ == 'ParticleHoleTransformer':
                ParticleHoleTransformer = transformer
            else:
                q_molecule = transformer.transform(q_molecule)

        if ParticleHoleTransformer is not None:
            q_molecule = ParticleHoleTransformer.transform(q_molecule)

        return q_molecule

    def _create_dipole_operators(self, q_molecule: QMolecule) -> \
            Tuple[FermionicOp, FermionicOp, FermionicOp]:
        x_dipole_operator = build_ferm_op_from_ints(q_molecule.x_dipole_integrals)
        y_dipole_operator = build_ferm_op_from_ints(q_molecule.y_dipole_integrals)
        z_dipole_operator = build_ferm_op_from_ints(q_molecule.z_dipole_integrals)

        return x_dipole_operator, y_dipole_operator, z_dipole_operator

    def _create_total_magnetization_operator(self, num_modes) -> FermionicOp:
        return build_ferm_op_from_ints(*calc_total_magnetization_ints(num_modes))

    def _create_total_angular_momentum_operator(self, num_modes) -> FermionicOp:
        return build_ferm_op_from_ints(*calc_total_ang_momentum_ints(num_modes))

    def _create_total_particle_number_operator(self, num_modes) -> FermionicOp:
        return build_ferm_op_from_ints(*calc_total_particle_num_ints(num_modes))
