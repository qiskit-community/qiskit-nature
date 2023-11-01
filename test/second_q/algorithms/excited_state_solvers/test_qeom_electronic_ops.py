# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
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

from qiskit_algorithms.utils import algorithm_globals

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper, TaperedQubitMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.algorithms.excited_states_solvers.qeom_electronic_ops_builder import (
    build_electronic_ops,
)
import qiskit_nature.optionals as _optionals
from .resources.expected_qeom_ops import (
    expected_hopping_operators_electronic,
    expected_commutativies_electronic,
    expected_indices_electronic,
)


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

        self.mapper = JordanWignerMapper()
        self.tapered_mapper = TaperedQubitMapper(JordanWignerMapper())
        self.electronic_structure_problem = self.driver.run()
        self.electronic_structure_problem.second_q_ops()

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built with a qubit mapper."""

        hopping_operators, commutativities, indices = build_electronic_ops(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            "sd",
            self.mapper,
        )

        with self.subTest("hopping operators"):
            self.assertEqual(hopping_operators.keys(), expected_hopping_operators_electronic.keys())
            for key, exp_key in zip(
                hopping_operators.keys(), expected_hopping_operators_electronic.keys()
            ):
                self.assertEqual(key, exp_key)
                val = hopping_operators[key]
                exp_val = expected_hopping_operators_electronic[exp_key]
                if not val.equiv(exp_val):
                    print(val)
                    print(exp_val)
                self.assertTrue(val.equiv(exp_val), msg=(val, exp_val))

        with self.subTest("commutativities"):
            self.assertEqual(commutativities, expected_commutativies_electronic)

        with self.subTest("excitation indices"):
            self.assertEqual(indices, expected_indices_electronic)

    def test_build_hopping_operators_taperedmapper(self):
        """Tests that the correct hopping operator is built with a tapered qubit mapper."""

        hopping_operators, commutativities, indices = build_electronic_ops(
            self.electronic_structure_problem.num_spatial_orbitals,
            self.electronic_structure_problem.num_particles,
            "sd",
            self.tapered_mapper,
        )

        with self.subTest("hopping operators"):
            self.assertEqual(hopping_operators.keys(), expected_hopping_operators_electronic.keys())
            for key, exp_key in zip(
                hopping_operators.keys(), expected_hopping_operators_electronic.keys()
            ):
                self.assertEqual(key, exp_key)
                val = hopping_operators[key]
                exp_val = expected_hopping_operators_electronic[exp_key]
                if not val.equiv(exp_val):
                    print(val)
                    print(exp_val)
                self.assertTrue(val.equiv(exp_val), msg=(val, exp_val))

        with self.subTest("commutativities"):
            self.assertEqual(commutativities, expected_commutativies_electronic)

        with self.subTest("excitation indices"):
            self.assertEqual(indices, expected_indices_electronic)


if __name__ == "__main__":
    unittest.main()
