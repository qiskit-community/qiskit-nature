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
from typing import List, Optional

from qiskit_nature.drivers import FermionicDriver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.molecular.fermionic_op_factory import create_fermionic_op
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

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations provided.

        Returns:
            A list of `SecondQuantizedOp`.
        """
        q_molecule = self.driver.run()
        for transformer in self.transformers:
            q_molecule = transformer.transform(q_molecule)
        fermionic_op = create_fermionic_op(q_molecule)

        return [SecondQuantizedOp([fermionic_op])]  # TODO support aux operators


