# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests ProteinShapeFileGen."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.results.protein_folding_tools.protein_shape_file_gen import ProteinShapeFileGen
import numpy as np


class TestProteinShapeFileGen(QiskitNatureTestCase):
    """Tests ProteinShapeFileGen."""

    test_xyz_1 = ProteinShapeFileGen(
        main_chain_turns=[1, 0, 3, 2, 0, 3],
        side_chain_turns=[None, None, None, None, None, None, None],
        peptide=Peptide("APRLRFY", [""] * 7),
    )

    test_xyz_2 = ProteinShapeFileGen(
        main_chain_turns=[1, 0, 3, 2],
        side_chain_turns=[None, None, 3, 3, None],
        peptide=Peptide("APRLR", ["", "", "F", "Y", ""]),
    )

    def test_side_positions(self):
        """Tests the side positions list."""
        expected1 = [None] * 7
        for result, expected in zip(self.test_xyz_1.side_positions, expected1):
            if expected is None:
                self.assertIsNone(result)
            else:
                np.testing.assert_almost_equal(result, expected, decimal=6)

        expected2 = [
            None,
            None,
            np.array([0.57735027, 0.57735027, -1.73205081]),
            np.array([2.30940108, -1.15470054, 0.0]),
            None,
        ]
        for result, expected in zip(self.test_xyz_2.side_positions, expected2):
            if expected is None:
                self.assertIsNone(result)
            else:
                np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_main_positions(self):
        """Tests the main position array."""
        expected1 = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.57735027, 0.57735027, -0.57735027],
                [1.15470054, 0.0, -1.15470054],
                [1.73205081, -0.57735027, -0.57735027],
                [2.30940108, 0.0, 0.0],
                [1.73205081, 0.57735027, 0.57735027],
                [1.15470054, 1.15470054, 0.0],
            ]
        )
        np.testing.assert_almost_equal(
            self.test_xyz_1.main_positions,
            expected1,
            decimal=6,
        )

        expected2 = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.57735027, 0.57735027, -0.57735027],
                [1.15470054, 0.0, -1.15470054],
                [1.73205081, -0.57735027, -0.57735027],
                [2.30940108, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(
            self.test_xyz_2.main_positions,
            expected2,
            decimal=6,
        )

    def test_xyz_file(self):
        """Tests the array that will get converted in a xyz file."""

        expected1 = np.array(
            [
                ["A", "0.0", "0.0", "0.0"],
                ["P", "0.5773502691896258", "0.5773502691896258", "-0.5773502691896258"],
                ["R", "1.1547005383792517", "0.0", "-1.1547005383792517"],
                ["L", "1.7320508075688776", "-0.5773502691896258", "-0.5773502691896258"],
                ["R", "2.3094010767585034", "0.0", "0.0"],
                ["F", "1.7320508075688776", "0.5773502691896258", "0.5773502691896258"],
                ["Y", "1.154700538379252", "1.1547005383792517", "0.0"],
            ],
            dtype="<U32",
        )

        np.testing.assert_equal(self.test_xyz_1.get_xyz_file(), expected1)

        expected2 = np.array(
            [
                ["A", "0.0", "0.0", "0.0"],
                ["P", "0.5773502691896258", "0.5773502691896258", "-0.5773502691896258"],
                ["R", "1.1547005383792517", "0.0", "-1.1547005383792517"],
                ["L", "1.7320508075688776", "-0.5773502691896258", "-0.5773502691896258"],
                ["R", "2.3094010767585034", "0.0", "0.0"],
                ["F", "0.5773502691896258", "0.5773502691896258", "-1.7320508075688776"],
                ["Y", "2.3094010767585034", "-1.1547005383792517", "0.0"],
            ],
            dtype="<U32",
        )

        np.testing.assert_array_equal(self.test_xyz_2.get_xyz_file(), expected2)
