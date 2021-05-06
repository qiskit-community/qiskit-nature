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

from qiskit_nature.circuit.library.ansatzes.utils.vibration_excitation_generator import (
    generate_vibration_excitations,
)


@ddt
class TestVibrationExcitationGenerator(QiskitNatureTestCase):
    """Tests for the default vibration excitation generator method."""

    @unpack
    @data(
        (1, [2], [((0,), (1,))]),
        (1, [3], [((0,), (1,)), ((0,), (2,))]),
        (2, [3], []),
        (1, [2, 2], [((0,), (1,)), ((2,), (3,))]),
        (2, [2, 2], [((0, 2), (1, 3))]),
        (1, [3, 3], [((0,), (1,)), ((0,), (2,)), ((3,), (4,)), ((3,), (5,))]),
        (
            2,
            [3, 3],
            [((0, 3), (1, 4)), ((0, 3), (1, 5)), ((0, 3), (2, 4)), ((0, 3), (2, 5))],
        ),
        (3, [3, 3], []),
        (2, [2, 2, 2], [((0, 2), (1, 3)), ((0, 4), (1, 5)), ((2, 4), (3, 5))]),
        (3, [2, 2, 2], [((0, 2, 4), (1, 3, 5))]),
        (4, [2, 2, 2], []),
        (2, [2, 3], [((0, 2), (1, 3)), ((0, 2), (1, 4))]),
        (
            2,
            [2, 3, 2],
            [
                ((0, 2), (1, 3)),
                ((0, 2), (1, 4)),
                ((0, 5), (1, 6)),
                ((2, 5), (3, 6)),
                ((2, 5), (4, 6)),
            ],
        ),
        (3, [2, 3, 2], [((0, 2, 5), (1, 3, 6)), ((0, 2, 5), (1, 4, 6))]),
    )
    def test_generate_excitations(self, num_excitations, num_modals, expect):
        """Test standard input arguments."""
        excitations = generate_vibration_excitations(num_excitations, num_modals)
        self.assertEqual(excitations, expect)
