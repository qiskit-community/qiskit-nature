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

"""Tests Hopping Operators builder."""
import unittest
from test import QiskitNatureTestCase

from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import QubitConverter, JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.algorithms.excited_states_solvers.qeom_electronic_ops_builder import (
    build_electronic_ops,
)
import qiskit_nature.optionals as _optionals


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.75",
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )

        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = self.driver.run()
        self.electronic_structure_problem.second_q_ops()

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built."""
        # TODO extract it somewhere
        expected_hopping_operators = {
            "E_0": PauliSumOp.from_list([("IIXY", -1j), ("IIYY", 1), ("IIXX", 1), ("IIYX", 1j)]),
            "Edag_0": PauliSumOp.from_list([("IIXY", 1j), ("IIXX", 1), ("IIYY", 1), ("IIYX", -1j)]),
            "E_1": PauliSumOp.from_list([("XYII", -1j), ("YYII", 1), ("XXII", 1), ("YXII", 1j)]),
            "Edag_1": PauliSumOp.from_list([("XYII", 1j), ("YYII", 1), ("XXII", 1), ("YXII", -1j)]),
            "E_2": PauliSumOp.from_list(
                [
                    ("XYXY", 1),
                    ("YYXY", 1j),
                    ("XYYY", 1j),
                    ("YYYY", -1),
                    ("XXXY", 1j),
                    ("YXXY", -1),
                    ("XXYY", -1),
                    ("YXYY", -1j),
                    ("XYXX", 1j),
                    ("YYXX", -1),
                    ("XYYX", -1),
                    ("YYYX", -1j),
                    ("XXXX", -1),
                    ("YXXX", -1j),
                    ("XXYX", -1j),
                    ("YXYX", 1),
                ]
            ),
            "Edag_2": PauliSumOp.from_list(
                [
                    ("XYXY", 1),
                    ("XXXY", -1j),
                    ("XYXX", -1j),
                    ("XXXX", -1),
                    ("YYXY", -1j),
                    ("YXXY", -1),
                    ("YYXX", -1),
                    ("YXXX", 1j),
                    ("XYYY", -1j),
                    ("XXYY", -1),
                    ("XYYX", -1),
                    ("XXYX", 1j),
                    ("YYYY", -1),
                    ("YXYY", 1j),
                    ("YYYX", 1j),
                    ("YXYX", 1),
                ]
            ),
        }
        expected_commutativies = {
            "E_0": [],
            "Edag_0": [],
            "E_1": [],
            "Edag_1": [],
            "E_2": [],
            "Edag_2": [],
        }
        expected_indices = {
            "E_0": ((0,), (1,)),
            "Edag_0": ((1,), (0,)),
            "E_1": ((2,), (3,)),
            "Edag_1": ((3,), (2,)),
            "E_2": ((0, 2), (1, 3)),
            "Edag_2": ((1, 3), (0, 2)),
        }

        hopping_operators, commutativities, indices = build_electronic_ops(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            "sd",
            self.qubit_converter,
        )

        with self.subTest("hopping operators"):
            self.assertEqual(hopping_operators.keys(), expected_hopping_operators.keys())
            for key, exp_key in zip(hopping_operators.keys(), expected_hopping_operators.keys()):
                self.assertEqual(key, exp_key)
                val = hopping_operators[key]
                exp_val = expected_hopping_operators[exp_key]
                if not val.equals(exp_val):
                    print(val)
                    print(exp_val)
                self.assertTrue(val.equals(exp_val))

        with self.subTest("commutativities"):
            self.assertEqual(commutativities, expected_commutativies)

        with self.subTest("excitation indices"):
            self.assertEqual(indices, expected_indices)


if __name__ == "__main__":
    unittest.main()
