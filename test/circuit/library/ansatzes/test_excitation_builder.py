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

from qiskit_nature.circuit.library.ansatzes.excitation_builder import build_excitation_ops
from qiskit_nature.operators.second_quantization import FermionicOp


@ddt
class TestExcitationBuilder(QiskitNatureTestCase):
    """TODO"""

    @unpack
    @data(
        (1, 4, [1, 1], [FermionicOp([('+-II', 1j), ('-+II', 1j)]),
                        FermionicOp([('II+-', 1j), ('II-+', 1j)])]),
        (1, 4, [2, 2], []),
        (1, 6, [1, 1], [FermionicOp([('+-IIII', 1j), ('-+IIII', 1j)]),
                        FermionicOp([('+I-III', 1j), ('-I+III', 1j)]),
                        FermionicOp([('III+-I', 1j), ('III-+I', 1j)]),
                        FermionicOp([('III+I-', 1j), ('III-I+', 1j)])]),
        (1, 6, [2, 2], [FermionicOp([('+I-III', 1j), ('-I+III', 1j)]),
                        FermionicOp([('I+-III', 1j), ('I-+III', 1j)]),
                        FermionicOp([('III+I-', 1j), ('III-I+', 1j)]),
                        FermionicOp([('IIII+-', 1j), ('IIII-+', 1j)])]),
        (1, 6, [3, 3], []),
        (2, 4, [1, 1], [FermionicOp([('+-+-', 1j), ('-+-+', -1j)])]),
        (2, 4, [2, 2], []),
        (2, 6, [1, 1], [FermionicOp([('+-I+-I', 1j), ('-+I-+I', -1j)]),
                        FermionicOp([('+-I+I-', 1j), ('-+I-I+', -1j)]),
                        FermionicOp([('+I-+-I', 1j), ('-I+-+I', -1j)]),
                        FermionicOp([('+I-+I-', 1j), ('-I+-I+', -1j)])]),
        (2, 6, [2, 2], [FermionicOp([('+I-+I-', 1j), ('-I+-I+', -1j)]),
                        FermionicOp([('+I-I+-', 1j), ('-I+I-+', -1j)]),
                        FermionicOp([('I+-+I-', 1j), ('I-+-I+', -1j)]),
                        FermionicOp([('I+-I+-', 1j), ('I-+I-+', -1j)])]),
        (2, 6, [3, 3], []),
        (2, 8, [2, 2], [FermionicOp([('++--IIII', 1j), ('--++IIII', -1j)]),
                        FermionicOp([('+I-I+I-I', 1j), ('-I+I-I+I', -1j)]),
                        FermionicOp([('+I-I+II-', 1j), ('-I+I-II+', -1j)]),
                        FermionicOp([('+I-II+-I', 1j), ('-I+II-+I', -1j)]),
                        FermionicOp([('+I-II+I-', 1j), ('-I+II-I+', -1j)]),
                        FermionicOp([('++--IIII', 1j), ('--++IIII', -1j)]),
                        FermionicOp([('+II-+I-I', 1j), ('-II+-I+I', -1j)]),
                        FermionicOp([('+II-+II-', 1j), ('-II+-II+', -1j)]),
                        FermionicOp([('+II-I+-I', 1j), ('-II+I-+I', -1j)]),
                        FermionicOp([('+II-I+I-', 1j), ('-II+I-I+', -1j)]),
                        FermionicOp([('I+-I+I-I', 1j), ('I-+I-I+I', -1j)]),
                        FermionicOp([('I+-I+II-', 1j), ('I-+I-II+', -1j)]),
                        FermionicOp([('I+-II+-I', 1j), ('I-+II-+I', -1j)]),
                        FermionicOp([('I+-II+I-', 1j), ('I-+II-I+', -1j)]),
                        FermionicOp([('I+I-+I-I', 1j), ('I-I+-I+I', -1j)]),
                        FermionicOp([('I+I-+II-', 1j), ('I-I+-II+', -1j)]),
                        FermionicOp([('I+I-I+-I', 1j), ('I-I+I-+I', -1j)]),
                        FermionicOp([('I+I-I+I-', 1j), ('I-I+I-I+', -1j)]),
                        FermionicOp([('IIII++--', 1j), ('IIII--++', -1j)]),
                        FermionicOp([('IIII++--', 1j), ('IIII--++', -1j)])]),
        (3, 8, [2, 1], [FermionicOp([('++--+-II', 1j), ('--++-+II', 1j)]),
                        FermionicOp([('++--+I-I', 1j), ('--++-I+I', 1j)]),
                        FermionicOp([('++--+II-', 1j), ('--++-II+', 1j)]),
                        FermionicOp([('++--+-II', 1j), ('--++-+II', 1j)]),
                        FermionicOp([('++--+I-I', 1j), ('--++-I+I', 1j)]),
                        FermionicOp([('++--+II-', 1j), ('--++-II+', 1j)])]),
    )
    def test_build_excitation_ops(self, num_excitations, num_spin_orbitals, num_particles, expect):
        """TODO"""
        ops = build_excitation_ops(num_excitations, num_spin_orbitals, num_particles)
        assert len(ops) == len(expect)
        for op, exp in zip(ops, expect):
            assert op._labels == exp._labels
            assert op._coeffs == exp._coeffs

    @unpack
    @data(
        (1, 4, [1, 1], 1, [FermionicOp([('+-II', 1j), ('-+II', 1j)]),
                           FermionicOp([('II+-', 1j), ('II-+', 1j)])]),
        (2, 4, [1, 1], 1, [FermionicOp([('+-+-', 1j), ('-+-+', -1j)])]),
        (1, 6, [1, 1], 1, [FermionicOp([('+-IIII', 1j), ('-+IIII', 1j)]),
                           FermionicOp([('+I-III', 1j), ('-I+III', 1j)]),
                           FermionicOp([('III+-I', 1j), ('III-+I', 1j)]),
                           FermionicOp([('III+I-', 1j), ('III-I+', 1j)])]),
        (2, 6, [1, 1], 1, [FermionicOp([('+-I+-I', 1j), ('-+I-+I', -1j)]),
                           FermionicOp([('+-I+I-', 1j), ('-+I-I+', -1j)]),
                           FermionicOp([('+I-+-I', 1j), ('-I+-+I', -1j)]),
                           FermionicOp([('+I-+I-', 1j), ('-I+-I+', -1j)])]),
        (2, 8, [2, 2], 1, [FermionicOp([('+I-I+I-I', 1j), ('-I+I-I+I', -1j)]),
                           FermionicOp([('+I-I+II-', 1j), ('-I+I-II+', -1j)]),
                           FermionicOp([('+I-II+-I', 1j), ('-I+II-+I', -1j)]),
                           FermionicOp([('+I-II+I-', 1j), ('-I+II-I+', -1j)]),
                           FermionicOp([('+II-+I-I', 1j), ('-II+-I+I', -1j)]),
                           FermionicOp([('+II-+II-', 1j), ('-II+-II+', -1j)]),
                           FermionicOp([('+II-I+-I', 1j), ('-II+I-+I', -1j)]),
                           FermionicOp([('+II-I+I-', 1j), ('-II+I-I+', -1j)]),
                           FermionicOp([('I+-I+I-I', 1j), ('I-+I-I+I', -1j)]),
                           FermionicOp([('I+-I+II-', 1j), ('I-+I-II+', -1j)]),
                           FermionicOp([('I+-II+-I', 1j), ('I-+II-+I', -1j)]),
                           FermionicOp([('I+-II+I-', 1j), ('I-+II-I+', -1j)]),
                           FermionicOp([('I+I-+I-I', 1j), ('I-I+-I+I', -1j)]),
                           FermionicOp([('I+I-+II-', 1j), ('I-I+-II+', -1j)]),
                           FermionicOp([('I+I-I+-I', 1j), ('I-I+I-+I', -1j)]),
                           FermionicOp([('I+I-I+I-', 1j), ('I-I+I-I+', -1j)])]),
    )
    def test_max_spin_excitation_ops(self, num_excitations, num_spin_orbitals, num_particles,
                                     max_spin, expect):
        """TODO"""
        ops = build_excitation_ops(num_excitations, num_spin_orbitals, num_particles,
                                   max_spin_excitation=max_spin)
        assert len(ops) == len(expect)
        for op, exp in zip(ops, expect):
            assert op._labels == exp._labels
            assert op._coeffs == exp._coeffs

    @unpack
    @data(
        (1, 4, [1, 1], [FermionicOp([('+-II', 1j), ('-+II', 1j)])]),
        (1, 6, [1, 1], [FermionicOp([('+-IIII', 1j), ('-+IIII', 1j)]),
                        FermionicOp([('+I-III', 1j), ('-I+III', 1j)])]),
        (2, 8, [2, 2], [FermionicOp([('++--IIII', 1j), ('--++IIII', -1j)]),
                        FermionicOp([('++--IIII', 1j), ('--++IIII', -1j)])]),
    )
    def test_pure_alpha_excitation_ops(self, num_excitations, num_spin_orbitals, num_particles,
                                       expect):
        """TODO"""
        ops = build_excitation_ops(num_excitations, num_spin_orbitals, num_particles,
                                   beta_spin=False)
        assert len(ops) == len(expect)
        for op, exp in zip(ops, expect):
            assert op._labels == exp._labels
            assert op._coeffs == exp._coeffs

    @unpack
    @data(
        (1, 4, [1, 1], [FermionicOp([('II+-', 1j), ('II-+', 1j)])]),
        (1, 6, [1, 1], [FermionicOp([('III+-I', 1j), ('III-+I', 1j)]),
                        FermionicOp([('III+I-', 1j), ('III-I+', 1j)])]),
        (2, 8, [2, 2], [FermionicOp([('IIII++--', 1j), ('IIII--++', -1j)]),
                        FermionicOp([('IIII++--', 1j), ('IIII--++', -1j)])]),
    )
    def test_pure_beta_excitation_ops(self, num_excitations, num_spin_orbitals, num_particles,
                                      expect):
        """TODO"""
        ops = build_excitation_ops(num_excitations, num_spin_orbitals, num_particles,
                                   alpha_spin=False)
        assert len(ops) == len(expect)
        for op, exp in zip(ops, expect):
            assert op._labels == exp._labels
            assert op._coeffs == exp._coeffs
