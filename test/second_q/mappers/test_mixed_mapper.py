# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Mixed Mapper """

import unittest


from test import QiskitNatureTestCase

from ddt import ddt, data, unpack
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp
from qiskit_nature.second_q.mappers import (
    BosonicLinearMapper,
    JordanWignerMapper,
    MixedMapper,
)
from qiskit_nature import settings


@ddt
class TestMixedMapper(QiskitNatureTestCase):
    """Test Mixed Mapper"""

    # Define some useful coefficients
    sq_2 = np.sqrt(2)

    bos_op1 = BosonicOp({"+_0": 1})
    mapped_bos_op1 = SparsePauliOp(
        ["XX", "YY", "YX", "XY"], coeffs=[0.25, 0.25, -0.25j, 0.25j]
    )

    bos_op2 = BosonicOp({"-_0": 1})
    mapped_bos_op2 = SparsePauliOp(
        ["XX", "YY", "YX", "XY"], coeffs=[0.25, 0.25, 0.25j, -0.25j]
    )

    fer_op1 = FermionicOp({"+_0": 1}, num_spin_orbitals=1)
    mapped_fer_op1 = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])

    fer_op2 = FermionicOp({"-_0": 1}, num_spin_orbitals=1)
    mapped_fer_op2 = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)])

    bos_op5 = BosonicOp({"+_0 -_0": 1})
    bos_op6 = BosonicOp({"-_0 +_0": 1})

    bos_mapper = BosonicLinearMapper(max_occupation=1)
    fer_mapper = JordanWignerMapper()
    hilbert_space_registers = {"b1": 2, "f1": 1}
    mappers = {"b1": bos_mapper, "f1": fer_mapper}
    mix_mapper = MixedMapper(mappers=mappers)

    def test_absolute_mapping(self):
        """Test the ``MixedOp`` mapping and compare to the calculated values."""

        target = self.mapped_bos_op1.tensor(3.0 * self.mapped_fer_op1)

        aux = settings.use_pauli_sum_op
        settings.use_pauli_sum_op = False
        comp_op1 = MixedOp({("b1", "f1"): [(3.0, self.bos_op1, self.fer_op1)]})
        test = self.mix_mapper.map(
            comp_op1, hilbert_space_registers=self.hilbert_space_registers
        )
        settings.use_pauli_sum_op = aux

        self.assertEqual(target, test)

    @data(
        (bos_op1, fer_op1, 2.0 + 1.0j),
        (bos_op2, fer_op2, 3.0 + 2.0j),
        (bos_op5, fer_op1, -4.0 - 3.0j),
        (bos_op6, fer_op2, -5.0 + 4j),
    )
    @unpack
    def test_relative_mapping(self, bos_op, fer_op, coef):
        """Test the ``MixedOp`` mapping and compare to the composition of the mapped operators."""

        composed_op = MixedOp({("b1", "f1"): [(coef, bos_op, fer_op)]})

        aux = settings.use_pauli_sum_op
        settings.use_pauli_sum_op = False
        target = coef * self.bos_mapper.map(bos_op).tensor(self.fer_mapper.map(fer_op))
        test = self.mix_mapper.map(
            composed_op, hilbert_space_registers=self.hilbert_space_registers
        )
        settings.use_pauli_sum_op = aux
        print(target)
        print(test)
        self.assertEqual(target, test)


if __name__ == "__main__":
    unittest.main()
