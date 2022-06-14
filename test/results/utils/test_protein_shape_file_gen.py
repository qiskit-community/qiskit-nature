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
import os
import filecmp
import tempfile
from test import QiskitNatureTestCase
import numpy as np
from ddt import ddt, data, unpack
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.results.utils.protein_shape_file_gen import ProteinShapeFileGen


@ddt
class TestProteinShapeFileGen(QiskitNatureTestCase):
    """Tests ProteinShapeFileGen."""

    @unpack
    @data(
        # First
        (
            [1, 0, 3, 2, 0, 3],
            [None, None, None, None, None, None, None],
            "APRLRFY",
            [""] * 7,
            [None] * 7,
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
            np.array(
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
            ),
            "Only_Main_Chain",
        ),
        # Second
        (
            [1, 0, 3, 2],
            [None, None, 3, 3, None],
            "APRLR",
            ["", "", "F", "Y", ""],
            [
                None,
                None,
                np.array([0.57735027, 0.57735027, -1.73205081]),
                np.array([2.30940108, -1.15470054, 0.0]),
                None,
            ],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.57735027, 0.57735027, -0.57735027],
                    [1.15470054, 0.0, -1.15470054],
                    [1.73205081, -0.57735027, -0.57735027],
                    [2.30940108, 0.0, 0.0],
                ]
            ),
            np.array(
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
            ),
            "Also_Side_Chains",
        ),
    )
    def test_shape(
        self,
        main_chain_turns,
        side_chain_turns,
        main_chain_residue_sequence,
        side_chain_residue_sequences,
        side_positions,
        main_positions,
        xyz,
        name_file,
    ):
        """Tests if ProteinShapeFileGen is properly initialized and its attributes are properly set."""
        peptide = Peptide(
            main_chain_residue_sequence=main_chain_residue_sequence,
            side_chain_residue_sequences=side_chain_residue_sequences,
        )
        filegen = ProteinShapeFileGen(
            main_chain_turns=main_chain_turns,
            side_chain_turns=side_chain_turns,
            peptide=peptide,
        )
        with self.subTest("Side Positions"):
            for result, expected in zip(filegen.side_positions, side_positions):
                if expected is None:
                    self.assertIsNone(result)
                else:
                    np.testing.assert_almost_equal(result, expected, decimal=6)

        with self.subTest("Main Positions"):
            np.testing.assert_almost_equal(
                filegen.main_positions,
                main_positions,
                decimal=6,
            )
        with self.subTest("XYZ file data"):
            np.testing.assert_equal(filegen.get_xyz_data(), xyz)

        with self.subTest("Write file"):
            current_dir = os.path.dirname(__file__)
            test_path = os.path.join(current_dir, "resources")
            file_test = os.path.join(test_path, name_file + "_test.xyz")
            with tempfile.TemporaryDirectory() as tmpdirname:
                filegen.save_xyz_file(
                    name=name_file + "_temp", path=tmpdirname, comment="This is a dummy comment."
                )

                file_temp = os.path.join(tmpdirname, name_file + "_temp.xyz")
                self.assertTrue(filecmp.cmp(file_temp, file_test))


