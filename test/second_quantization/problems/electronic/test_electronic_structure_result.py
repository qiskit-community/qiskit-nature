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

"""Tests for the ElectronicStructureResult."""

import contextlib
import io
from itertools import zip_longest
from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.second_quantization.results import ElectronicStructureResult


class TestElectronicStructureResult(QiskitNatureTestCase):
    # pylint: disable=attribute-defined-outside-init
    """Additional tests asserting some edge cases of the ElectronicStructureResult."""

    def _assert_printed_result(self, result):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            print(result)
        for truth, expected in zip_longest(out.getvalue().split("\n"), self.expected.split("\n")):
            if expected is None:
                return
            assert truth.strip().startswith(expected.strip())

    def test_print_empty(self):
        """Test printing an empty result."""
        res = ElectronicStructureResult()
        self.expected = """\
            === GROUND STATE ENERGY ===
        """
        self._assert_printed_result(res)

    def test_print_complex(self):
        """Test printing complex numbers."""
        res = ElectronicStructureResult()
        res.computed_energies = np.asarray([1.0j])
        self.expected = """\
            === GROUND STATE ENERGY ===

            * Electronic ground state energy (Hartree): 0.0+1.j
              - computed part:      0.0+1.j
        """
        self._assert_printed_result(res)

    def test_print_complex_dipole(self):
        """Test printing complex dipoles."""
        res = ElectronicStructureResult()
        res.computed_energies = np.asarray([1.0])
        res.nuclear_dipole_moment = (0.0, 0.0, 1.0)
        res.computed_dipole_moment = [(0.0, 0.0, 1.0j)]
        res.extracted_transformer_dipoles = [{}]
        self.expected = """\
            === GROUND STATE ENERGY ===

            * Electronic ground state energy (Hartree): 1.
              - computed part:      1.

            === DIPOLE MOMENTS ===

            ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.]

              0:
              * Electronic dipole moment (a.u.): [0.0  0.0  0.0+1.j]
                - computed part:      [0.0  0.0  0.0+1.j]
              > Dipole moment (a.u.): [0.0  0.0  1.+1.j]  Total: 1.+1.j
                             (debye): [0.0  0.0  2.54174623+2.54174623j]  Total: 2.54174623+2.54174623j
        """
        self._assert_printed_result(res)
