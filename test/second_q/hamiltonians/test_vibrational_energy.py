# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test VibrationalEnergy Property"""

import json
import unittest
from test.second_q.properties.property_test import PropertyTest

from qiskit_nature.second_q.formats.watson import WatsonHamiltonian
from qiskit_nature.second_q.formats.watson_translator import watson_to_problem
from qiskit_nature.second_q.operators import VibrationalOp
from qiskit_nature.second_q.problems import HarmonicBasis
import qiskit_nature.optionals as _optionals


class TestVibrationalEnergy(PropertyTest):
    """Test VibrationalEnergy Property"""

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def setUp(self):
        """Setup basis."""
        super().setUp()
        basis = HarmonicBasis([2, 2, 2, 2])

        import sparse as sp  # pylint: disable=import-error

        watson = WatsonHamiltonian(
            quadratic_force_constants=sp.as_coo(
                {
                    (1, 1): 352.3005875,
                    (0, 0): 631.6153975,
                    (3, 3): 115.653915,
                    (2, 2): 115.653915,
                },
                shape=(4, 4),
            ),
            cubic_force_constants=sp.as_coo(
                {
                    (1, 1, 1): -15.341901966295344,
                    (0, 0, 1): -88.2017421687633,
                    (3, 3, 1): 42.40478531359112,
                    (3, 2, 1): 26.25167512727164,
                    (2, 2, 1): 2.2874639206341865,
                },
                shape=(4, 4, 4),
            ),
            quartic_force_constants=sp.as_coo(
                {
                    (1, 1, 1, 1): 0.4207357291666667,
                    (0, 0, 1, 1): 4.9425425,
                    (0, 0, 0, 0): 1.6122932291666665,
                    (3, 3, 1, 1): -4.194299375,
                    (2, 2, 1, 1): -4.194299375,
                    (3, 3, 0, 0): -10.20589125,
                    (2, 2, 0, 0): -10.20589125,
                    (3, 3, 3, 3): 2.2973803125,
                    (3, 3, 3, 2): 2.7821204166666664,
                    (3, 3, 2, 2): 7.329224375,
                    (3, 2, 2, 2): -2.7821200000000004,
                    (2, 2, 2, 2): 2.2973803125,
                },
                shape=(4, 4, 4, 4),
            ),
            kinetic_coefficients=sp.as_coo(
                {
                    (1, 1): -352.3005875,
                    (0, 0): -631.6153975,
                    (3, 3): -115.653915,
                    (2, 2): -115.653915,
                },
                shape=(4, 4),
            ),
        )

        problem = watson_to_problem(watson, basis)
        self.prop = problem.hamiltonian

    def test_second_q_op(self):
        """Test second_q_op."""
        op = self.prop.second_q_op().normal_order()
        with open(
            self.get_resource_path("vibrational_energy_op.json", "second_q/properties/resources"),
            "r",
            encoding="utf8",
        ) as file:
            expected = VibrationalOp(json.load(file)).normal_order()

        self.assertTrue(op.equiv(expected))


if __name__ == "__main__":
    unittest.main()
