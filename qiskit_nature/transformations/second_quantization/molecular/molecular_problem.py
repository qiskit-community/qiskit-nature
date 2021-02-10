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
from typing import List, Optional

from qiskit_nature import QMolecule
from qiskit_nature.drivers import FermionicDriver
from qiskit_nature.operators import FermionicOp
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
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
        fermionic_op = self.__create_fermionic_op(q_molecule)

        return [SecondQuantizedOp([fermionic_op])]  # TODO support aux operators

    def __create_fermionic_op(self, q_molecule: QMolecule) -> FermionicOp:
        one_body_ints = q_molecule.one_body_integrals
        two_body_ints = q_molecule.two_body_integrals
        fermionic_op = FermionicOp('I' * len(one_body_ints))
        fermionic_op = self.__populate_fermionic_op_with_one_body_integrals(fermionic_op, one_body_ints)
        fermionic_op = self.__populate_fermionic_op_with_two_body_integrals(fermionic_op, two_body_ints)

        fermionic_op = fermionic_op.reduce()

        return fermionic_op

    # TODO might likely be extracted to a separate module
    @staticmethod
    def __populate_fermionic_op_with_one_body_integrals(fermionic_op: FermionicOp, one_body_integrals):
        for idx in itertools.product(range(len(one_body_integrals)), repeat=2):
            coeff = one_body_integrals[idx]
            if not coeff:
                continue
            label = ['I'] * len(one_body_integrals)
            base_op = coeff * FermionicOp(''.join(label))
            for i, op in [(idx[0], '+'), (idx[1], '-')]:
                label_i = label.copy()
                label_i[i] = op
                base_op @= FermionicOp(''.join(label_i))
            fermionic_op += base_op
        return fermionic_op

    # TODO might likely be extracted to a separate module
    @staticmethod
    def __populate_fermionic_op_with_two_body_integrals(fermionic_op: FermionicOp, two_body_integrals):
        for idx in itertools.product(range(len(two_body_integrals)), repeat=4):
            coeff = two_body_integrals[idx]
            if not coeff:
                continue
            label = ['I'] * len(two_body_integrals)
            base_op = coeff * FermionicOp(''.join(label))
            for i, op in [(idx[0], '+'), (idx[2], '+'), (idx[3], '-'), (idx[1], '-')]:
                label_i = label.copy()
                label_i[i] = op
                base_op @= FermionicOp(''.join(label_i))
            base_op.reduce()
            fermionic_op += base_op
        return fermionic_op
