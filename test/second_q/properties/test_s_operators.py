# This code is part of a Qiskit project.
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

"""Test the spin operator generator functions."""

import unittest
from test import QiskitNatureTestCase

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators.commutators import commutator
from qiskit_nature.second_q.properties import AngularMomentum
from qiskit_nature.second_q.properties.s_operators import (
    s_minus_operator,
    s_plus_operator,
    s_x_operator,
    s_y_operator,
    s_z_operator,
)


class TestSOperators(QiskitNatureTestCase):
    """Tests for the spin operator generator functions."""

    def test_s_plus_operator(self) -> None:
        """Tests the $S^+$ operator directly."""
        truth = {f"+_{i} -_{i+4}": 1.0 for i in range(4)}
        s_p = s_plus_operator(4)
        self.assertEqual(s_p, FermionicOp(truth))

    def test_s_minus_operator(self) -> None:
        """Tests the $S^-$ operator directly."""
        truth = {f"+_{i+4} -_{i}": 1.0 for i in range(4)}
        s_m = s_minus_operator(4)
        self.assertEqual(s_m, FermionicOp(truth))

    def test_s_z_operator(self) -> None:
        """Tests the $S^z$ operator directly."""
        truth = {f"+_{i} -_{i}": 0.5 for i in range(4)}
        truth.update({f"+_{i} -_{i}": -0.5 for i in range(4, 8)})
        s_z = s_z_operator(4)
        self.assertEqual(s_z, FermionicOp(truth))

    def test_s_x_relation(self) -> None:
        """Tests that :math:`S^x = 0.5 * (S^+ + S^-)`."""
        s_x = s_x_operator(4)
        s_p = s_plus_operator(4)
        s_m = s_minus_operator(4)
        self.assertEqual(s_x, 0.5 * (s_p + s_m))

    def test_s_y_relation(self) -> None:
        """Tests that :math:`S^y = -0.5j * (S^+ - S^-)`."""
        s_y = s_y_operator(4)
        s_p = s_plus_operator(4)
        s_m = s_minus_operator(4)
        self.assertEqual(s_y, -0.5j * (s_p - s_m))

    def test_commutator_xyz(self) -> None:
        """Tests that :math:`[S^x, S^y] = 1j * S^z`."""
        s_x = s_x_operator(4)
        s_y = s_y_operator(4)
        s_z = s_z_operator(4)
        self.assertEqual(commutator(s_x, s_y), 1j * s_z)

    def test_commutator_yzx(self) -> None:
        """Tests that :math:`[S^y, S^z] = 1j * S^x`."""
        s_x = s_x_operator(4)
        s_y = s_y_operator(4)
        s_z = s_z_operator(4)
        self.assertEqual(commutator(s_y, s_z), 1j * s_x)

    def test_commutator_zxy(self) -> None:
        """Tests that :math:`[S^z, S^x] = 1j * S^y`."""
        s_x = s_x_operator(4)
        s_y = s_y_operator(4)
        s_z = s_z_operator(4)
        self.assertEqual(commutator(s_z, s_x), 1j * s_y)

    def test_commutator_s2x(self) -> None:
        """Tests that :math:`[S^2, S^x] = 0`."""
        s_x = s_x_operator(4)
        s_2 = AngularMomentum(4).second_q_ops()["AngularMomentum"]
        self.assertEqual(commutator(s_2, s_x), FermionicOp.zero())

    def test_commutator_s2y(self) -> None:
        """Tests that :math:`[S^2, S^y] = 0`."""
        s_y = s_y_operator(4)
        s_2 = AngularMomentum(4).second_q_ops()["AngularMomentum"]
        self.assertEqual(commutator(s_2, s_y), FermionicOp.zero())

    def test_commutator_s2z(self) -> None:
        """Tests that :math:`[S^2, S^z] = 0`."""
        s_z = s_z_operator(4)
        s_2 = AngularMomentum(4).second_q_ops()["AngularMomentum"]
        self.assertEqual(commutator(s_2, s_z), FermionicOp.zero())


if __name__ == "__main__":
    unittest.main()
