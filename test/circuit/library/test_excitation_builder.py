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

"""Test the excitation builder."""

from test import QiskitNatureTestCase

from ddt import data, ddt, unpack

from qiskit_nature.circuit.library.ansaetze import ExcitationBuilder
from qiskit_nature.operators.second_quantization import FermionicOp, SecondQuantizedOp


@ddt
class TestExcitationBuilder(QiskitNatureTestCase):
    """TODO"""

    @unpack
    @data(
        (1, 4, [1, 1], [SecondQuantizedOp([FermionicOp('II-+')]),
                        SecondQuantizedOp([FermionicOp('-+II')])]),
        (1, 4, [2, 2], []),
        (1, 6, [1, 1], [SecondQuantizedOp([FermionicOp('III-+I')]),
                        SecondQuantizedOp([FermionicOp('III-I+')]),
                        SecondQuantizedOp([FermionicOp('-I+III')]),
                        SecondQuantizedOp([FermionicOp('-+IIII')])]),
        (1, 6, [2, 2], [SecondQuantizedOp([FermionicOp(('-I+III', 1))]),
                        SecondQuantizedOp([FermionicOp(('IIII-+', 1))]),
                        SecondQuantizedOp([FermionicOp(('I-+III', 1))]),
                        SecondQuantizedOp([FermionicOp(('III-I+', 1))])]),
        (1, 6, [3, 3], []),
        (2, 4, [1, 1], [SecondQuantizedOp([FermionicOp('-+-+')])]),
        (2, 4, [2, 2], []),
        (2, 6, [1, 1], [SecondQuantizedOp([FermionicOp('-+I-+I')]),
                        SecondQuantizedOp([FermionicOp('-I+-I+')]),
                        SecondQuantizedOp([FermionicOp('-+I-I+')]),
                        SecondQuantizedOp([FermionicOp('-I+-+I')])]),
        (2, 6, [2, 2], [SecondQuantizedOp([FermionicOp('I-+-I+')]),
                        SecondQuantizedOp([FermionicOp('-I+I-+')]),
                        SecondQuantizedOp([FermionicOp('-I+-I+')]),
                        SecondQuantizedOp([FermionicOp('I-+I-+')])]),
        (2, 6, [3, 3], []),
        (3, 8, [2, 1], [SecondQuantizedOp([FermionicOp(('--++-I+I', 1))]),
                        SecondQuantizedOp([FermionicOp(('--++-+II', 1))]),
                        SecondQuantizedOp([FermionicOp(('--++-+II', 1))]),
                        SecondQuantizedOp([FermionicOp(('--++-II+', 1))]),
                        SecondQuantizedOp([FermionicOp(('--++-II+', 1))]),
                        SecondQuantizedOp([FermionicOp(('--++-I+I', 1))])]),
    )
    def test_build_excitation_ops(self, num_excitations, num_spin_orbitals, num_particles, expect):
        """TODO"""
        ops = ExcitationBuilder.build_excitation_ops(num_excitations, num_spin_orbitals,
                                                     num_particles)
        for op, exp in zip(ops, expect):
            assert op.fermion._labels == exp.fermion._labels
            assert op.fermion._coeffs == exp.fermion._coeffs
