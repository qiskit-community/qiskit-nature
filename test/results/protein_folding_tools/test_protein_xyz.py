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
"""Tests ProteinXYZ."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.results.protein_folding_tools.protein_xyz import ProteinXYZ
import numpy as np


class TestProteinXYZ(QiskitNatureTestCase):
    """Tests ProteinXYZ."""

    test_xyz_1 = ProteinXYZ(
        main_chain_turns=[1, 0, 3, 2, 0, 3],
        side_chain_turns=[None, None, None, None, None, None, None],
        peptide=Peptide("APRLRFY", [""] * 7),
    )

    test_xyz_2 = ProteinXYZ(
        main_chain_turns=[1, 0, 3, 2],
        side_chain_turns=[None, None, 3, 3, None],
        peptide=Peptide("APRLR", ["", "", "F", "Y", ""]),
    )

    def test_side_positions(self):
        """Tests the side positions list."""
        for result, expected in zip(
            self.test_xyz_1.side_positions, [None, None, None, None, None, None, None]
        ):
            if expected is None:
                self.assertIsNone(result)
            else:
                self.assertTrue(np.allclose(result, expected, atol=1e-6))

        for result, expected in zip(
            self.test_xyz_2.side_positions,
            [
                None,
                None,
                np.array([0.57735027, 0.57735027, -1.73205081]),
                np.array([2.30940108, -1.15470054, 0.0]),
                None,
            ],
        ):
            if expected is None:
                self.assertIsNone(result)
            else:
                self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_main_positions(self):
        """Tests the main position array."""
        self.assertTrue(
            np.allclose(
                self.test_xyz_1.main_positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.57735027, 0.57735027, -0.57735027],
                        [1.15470054, 0.0, -1.15470054],
                        [1.73205081, -0.57735027, -0.57735027],
                        [2.30940108, 0.0, 0.0],
                        [1.73205081, 0.57735027, 0.57735027],
                        [1.15470054, 1.15470054, 0.0],
                    ]
                ),
                atol=1e-6,
            )
        )

        self.assertTrue(
            np.allclose(
                self.test_xyz_2.main_positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.57735027, 0.57735027, -0.57735027],
                        [1.15470054, 0.0, -1.15470054],
                        [1.73205081, -0.57735027, -0.57735027],
                        [2.30940108, 0.0, 0.0],
                    ]
                ),
                atol=1e-6,
            )
        )

    def test_xyz_file(self):
        """Tests the array that will get converted in a xyz file."""
        self.assertTrue(
            (
                self.test_xyz_1.get_xyz_file("", False)
                == np.array(
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
            ).all()
        )

        self.assertTrue(
            (
                self.test_xyz_2.get_xyz_file("", False)
                == np.array(
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
            ).all()
        )
