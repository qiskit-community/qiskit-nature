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

from qiskit_nature import WatsonHamiltonian
from qiskit_nature.drivers import BosonicDriver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.base_problem import BaseProblem
from qiskit_nature.problems.second_quantization.vibrational.spin_op_builder import build_spin_op
from qiskit_nature.transformers import BaseTransformer


class VibrationalProblem(BaseProblem):
    """Vibrational Problem"""

    def __init__(self, bosonic_driver: BosonicDriver,
                 transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            bosonic_driver: A bosonic driver encoding the molecule information.
            transformers: A list of transformations to be applied to the molecule.
        """
        super().__init__(bosonic_driver, transformers)

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: ... .
        """
        watson_hamiltonian = self.driver.run()
        watson_hamiltonian_transformed = self._transform(watson_hamiltonian)
        basis_size = 1  # TODO how to get it?
        truncation_order = 3  # TODO how to get it?
        bosonic_op = build_spin_op(watson_hamiltonian_transformed, basis_size, truncation_order)

        second_quantized_ops_list = [SecondQuantizedOp([bosonic_op])]

        return second_quantized_ops_list

    def _transform(self, watson_hamiltonian: WatsonHamiltonian) -> WatsonHamiltonian:
        for transformer in self.transformers:
            watson_hamiltonian = transformer.transform(watson_hamiltonian)
        return watson_hamiltonian
