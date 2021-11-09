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

"""Test the excitation generator."""

from test import QiskitNatureTestCase

from ddt import data, ddt, unpack

from qiskit_nature.circuit.library.ansatzes.utils.fermionic_excitation_generator import (
    generate_fermionic_excitations,
)


@ddt
class TestFermionicExcitationGenerator(QiskitNatureTestCase):
    """Tests for the default fermionic excitation generator method."""

    @unpack
    @data(
        (1, 4, [1, 1], [((0,), (1,)), ((2,), (3,))]),
        (1, 4, [2, 2], []),
        (1, 6, [1, 1], [((0,), (1,)), ((0,), (2,)), ((3,), (4,)), ((3,), (5,))]),
        (1, 6, [2, 2], [((0,), (2,)), ((1,), (2,)), ((3,), (5,)), ((4,), (5,))]),
        (1, 6, [3, 3], []),
        (2, 4, [1, 1], [((0, 2), (1, 3))]),
        (2, 4, [2, 2], []),
        (
            2,
            6,
            [1, 1],
            [((0, 3), (1, 4)), ((0, 3), (1, 5)), ((0, 3), (2, 4)), ((0, 3), (2, 5))],
        ),
        (
            2,
            6,
            [2, 2],
            [((0, 3), (2, 5)), ((0, 4), (2, 5)), ((1, 3), (2, 5)), ((1, 4), (2, 5))],
        ),
        (2, 6, [3, 3], []),
        (
            2,
            8,
            [2, 2],
            [
                ((0, 1), (2, 3)),
                ((0, 4), (2, 6)),
                ((0, 4), (2, 7)),
                ((0, 5), (2, 6)),
                ((0, 5), (2, 7)),
                ((0, 4), (3, 6)),
                ((0, 4), (3, 7)),
                ((0, 5), (3, 6)),
                ((0, 5), (3, 7)),
                ((1, 4), (2, 6)),
                ((1, 4), (2, 7)),
                ((1, 5), (2, 6)),
                ((1, 5), (2, 7)),
                ((1, 4), (3, 6)),
                ((1, 4), (3, 7)),
                ((1, 5), (3, 6)),
                ((1, 5), (3, 7)),
                ((4, 5), (6, 7)),
            ],
        ),
        (
            3,
            8,
            [2, 1],
            [((0, 1, 4), (2, 3, 5)), ((0, 1, 4), (2, 3, 6)), ((0, 1, 4), (2, 3, 7))],
        ),
    )
    def test_generate_excitations(self, num_excitations, num_spin_orbitals, num_particles, expect):
        """Test standard input arguments."""
        excitations = generate_fermionic_excitations(
            num_excitations, num_spin_orbitals, num_particles
        )
        self.assertEqual(excitations, expect)

    @unpack
    @data(
        (1, 4, [1, 1], 1, [((0,), (1,)), ((2,), (3,))]),
        (2, 4, [1, 1], 1, [((0, 2), (1, 3))]),
        (1, 6, [1, 1], 1, [((0,), (1,)), ((0,), (2,)), ((3,), (4,)), ((3,), (5,))]),
        (
            2,
            6,
            [1, 1],
            1,
            [((0, 3), (1, 4)), ((0, 3), (1, 5)), ((0, 3), (2, 4)), ((0, 3), (2, 5))],
        ),
        (
            2,
            8,
            [2, 2],
            1,
            [
                ((0, 4), (2, 6)),
                ((0, 4), (2, 7)),
                ((0, 5), (2, 6)),
                ((0, 5), (2, 7)),
                ((0, 4), (3, 6)),
                ((0, 4), (3, 7)),
                ((0, 5), (3, 6)),
                ((0, 5), (3, 7)),
                ((1, 4), (2, 6)),
                ((1, 4), (2, 7)),
                ((1, 5), (2, 6)),
                ((1, 5), (2, 7)),
                ((1, 4), (3, 6)),
                ((1, 4), (3, 7)),
                ((1, 5), (3, 6)),
                ((1, 5), (3, 7)),
            ],
        ),
    )
    def test_max_spin_excitation(
        self, num_excitations, num_spin_orbitals, num_particles, max_spin, expect
    ):
        """Test limiting the maximum number of excitations per spin species."""
        excitations = generate_fermionic_excitations(
            num_excitations,
            num_spin_orbitals,
            num_particles,
            max_spin_excitation=max_spin,
        )
        self.assertEqual(excitations, expect)

    @unpack
    @data(
        (1, 4, [1, 1], [((0,), (1,))]),
        (1, 6, [1, 1], [((0,), (1,)), ((0,), (2,))]),
        (2, 8, [2, 2], [((0, 1), (2, 3))]),
    )
    def test_pure_alpha_excitation(self, num_excitations, num_spin_orbitals, num_particles, expect):
        """Test disabling beta-spin excitations."""
        excitations = generate_fermionic_excitations(
            num_excitations, num_spin_orbitals, num_particles, beta_spin=False
        )
        self.assertEqual(excitations, expect)

    @unpack
    @data(
        (1, 4, [1, 1], [((2,), (3,))]),
        (1, 6, [1, 1], [((3,), (4,)), ((3,), (5,))]),
        (2, 8, [2, 2], [((4, 5), (6, 7))]),
    )
    def test_pure_beta_excitation(self, num_excitations, num_spin_orbitals, num_particles, expect):
        """Test disabling alpha-spin excitations."""
        excitations = generate_fermionic_excitations(
            num_excitations, num_spin_orbitals, num_particles, alpha_spin=False
        )
        self.assertEqual(excitations, expect)

    @unpack
    @data(
        (1, 4, [0, 0], [((0,), (1,)), ((2,), (3,))]),
        (1, 4, [1, 0], [((0,), (1,)), ((2,), (3,))]),
        (1, 4, [1, 1], [((0,), (1,)), ((2,), (3,))]),
        (2, 4, [0, 0], [((0, 2), (1, 3))]),
    )
    def test_generalized_excitations(
        self, num_excitations, num_spin_orbitals, num_particles, expect
    ):
        """Test generalized excitations."""
        excitations = generate_fermionic_excitations(
            num_excitations, num_spin_orbitals, num_particles, generalized=True
        )
        self.assertEqual(excitations, expect)

    @unpack
    @data(
        (1, 4, [1, 1], [((0,), (1,)), ((0,), (3,)), ((2,), (1,)), ((2,), (3,))]),
        (2, 4, [1, 1], [((0, 2), (1, 3))]),
        (
            2,
            6,
            [1, 1],
            [
                ((0, 3), (1, 2)),
                ((0, 3), (1, 4)),
                ((0, 3), (1, 5)),
                ((0, 3), (2, 4)),
                ((0, 3), (2, 5)),
                ((0, 3), (4, 5)),
            ],
        ),
    )
    def test_spin_flip_excitations(self, num_excitations, num_spin_orbitals, num_particles, expect):
        """Test allowing spin-flipped excitations."""
        excitations = generate_fermionic_excitations(
            num_excitations, num_spin_orbitals, num_particles, spin_flip=True
        )
        self.assertEqual(excitations, expect)
