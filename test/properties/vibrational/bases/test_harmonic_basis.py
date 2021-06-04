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

"""Test HarmonicBasis."""

from ddt import ddt, data, unpack
from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.vibrational.bases import HarmonicBasis
from qiskit_nature.properties.vibrational.integrals import VibrationalIntegrals


@ddt
class TestHarmonicBasis(QiskitNatureTestCase):
    """Test HarmonicBasis."""

    # TODO: extract data
    @unpack
    @data(
        (
            1,
            [
                (352.3005875, (2, 2)),
                (-352.3005875, (-2, -2)),
                (631.6153975, (1, 1)),
                (-631.6153975, (-1, -1)),
                (115.653915, (4, 4)),
                (-115.653915, (-4, -4)),
                (115.653915, (3, 3)),
                (-115.653915, (-3, -3)),
                (-15.341901966295344, (2, 2, 2)),
                (0.4207357291666667, (2, 2, 2, 2)),
                (1.6122932291666665, (1, 1, 1, 1)),
                (2.2973803125, (4, 4, 4, 4)),
                (2.2973803125, (3, 3, 3, 3)),
            ],
            (
                [
                    [0, 0, 1, 1, 1, 1, 2, 2, 3, 3],
                    [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                ],
                [
                    1268.06767469,
                    3813.87678344,
                    705.86338219,
                    -46.0257059,
                    -46.0257059,
                    2120.11456094,
                    238.19997094,
                    728.38419469,
                    238.19997094,
                    728.38419469,
                ],
            ),
        ),
        (
            2,
            [
                (-88.2017421687633, (1, 1, 2)),
                (42.40478531359112, (4, 4, 2)),
                (2.2874639206341865, (3, 3, 2)),
                (4.9425425, (1, 1, 2, 2)),
                (-4.194299375, (4, 4, 2, 2)),
                (-4.194299375, (3, 3, 2, 2)),
                (-10.20589125, (4, 4, 1, 1)),
                (-10.20589125, (3, 3, 1, 1)),
                (2.7821204166666664, (4, 4, 4, 3)),
                (7.329224375, (4, 4, 3, 3)),
                (-2.7821200000000004, (4, 3, 3, 3)),
            ],
            (
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                    ],
                    [
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                    ],
                    [
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                ],
                [
                    4.94254250e00,
                    -8.82017422e01,
                    -8.82017422e01,
                    1.48276275e01,
                    1.48276275e01,
                    -2.64605227e02,
                    -2.64605227e02,
                    4.44828825e01,
                    -1.02058913e01,
                    -3.06176738e01,
                    -4.19429938e00,
                    2.28746392e00,
                    2.28746392e00,
                    -1.25828981e01,
                    -3.06176738e01,
                    -9.18530213e01,
                    -1.25828981e01,
                    6.86239176e00,
                    6.86239176e00,
                    -3.77486944e01,
                    -1.02058913e01,
                    -3.06176738e01,
                    -4.19429938e00,
                    4.24047853e01,
                    4.24047853e01,
                    -1.25828981e01,
                    7.32922438e00,
                    2.19876731e01,
                    1.25000000e-06,
                    1.25000000e-06,
                    1.25000000e-06,
                    1.25000000e-06,
                    -3.06176738e01,
                    -9.18530213e01,
                    -1.25828981e01,
                    1.27214356e02,
                    1.27214356e02,
                    -3.77486944e01,
                    2.19876731e01,
                    6.59630194e01,
                ],
            ),
        ),
        (
            3,
            [
                (26.25167512727164, (4, 3, 2)),
            ],
            (
                [
                    [3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                ],
                [
                    26.25167513,
                    26.25167513,
                    26.25167513,
                    26.25167513,
                    26.25167513,
                    26.25167513,
                    26.25167513,
                    26.25167513,
                ],
            ),
        ),
    )
    def test_harmonic_basis(self, num_body, integrals, expected):
        """Test HarmonicBasis"""
        integrals = VibrationalIntegrals(num_body, integrals)

        # TODO: test more variants
        num_modes = 4
        num_modals = 2
        num_modals_per_mode = [num_modals] * num_modes
        basis = HarmonicBasis(num_modals_per_mode)

        integrals.basis = basis
        matrix = integrals.to_basis()
        nonzero = np.nonzero(matrix)

        exp_nonzero, exp_values = expected
        assert np.allclose(np.asarray(nonzero), np.asarray(exp_nonzero))
        assert np.allclose(matrix[nonzero], np.asarray(exp_values))
