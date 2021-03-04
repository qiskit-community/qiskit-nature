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
"""The Vibrational Problem class."""
from typing import List, Tuple, Optional

from qiskit_nature.drivers.qmolecule import QMolecule
from qiskit_nature.drivers import BosonicDriver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.transformers import BaseTransformer


class VibrationalProblem:
    """Vibrational Problem"""

    def __init__(self, bosonic_driver: BosonicDriver,
                 transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            bosonic_driver: A bosonic driver encoding the molecule information.
            transformers: A list of transformations to be applied to the molecule.
        """
        if transformers is None:
            transformers = []
        self.driver = bosonic_driver
        self.transformers = transformers

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: ... .
        """
        hamiltonian = self.driver.run()
        hamiltonian_transformed = self._transform_hamiltonian(hamiltonian)
        num_modes = hamiltonian_transformed.one_body_integrals.shape[0]

        second_quantized_ops_list = []

        return second_quantized_ops_list

    def _transform_hamiltonian(self, q_molecule) -> QMolecule:
        for transformer in self.transformers:
            q_molecule = transformer.transform(q_molecule)
        return q_molecule
