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

"""Tests Fermionic Operator builder."""
from test import QiskitNatureTestCase
from test.problems.second_quantization.electronic.resources.resource_reader import (
    read_expected_file,
)
import numpy as np
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.problems.second_quantization.electronic.builders import (
    fermionic_op_builder,
)
from qiskit_nature.drivers import HDF5Driver


class TestFermionicOpBuilder(QiskitNatureTestCase):
    """Tests Fermionic Operator builder."""

    def test_build_fermionic_op(self):
        """Tests that the correct FermionicOp is built from QMolecule."""
        expected_num_of_terms_ferm_op = 184
        expected_fermionic_op_path = self.get_resource_path(
            "H2_631g_ferm_op_two_ints",
            "problems/second_quantization/" "electronic/resources",
        )
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_631g.hdf5", "transformers")
        )
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder._build_fermionic_op(q_molecule)
        with self.subTest("Check type of fermionic operator"):
            assert isinstance(fermionic_op, FermionicOp)
        with self.subTest("Check expected number of terms in a fermionic operator."):
            assert len(fermionic_op) == expected_num_of_terms_ferm_op
        with self.subTest("Check expected content of a fermionic operator."):
            assert all(
                s[0] == t[0] and np.isclose(s[1], t[1])
                for s, t in zip(fermionic_op.to_list(), expected_fermionic_op)
            )

    def test_build_fermionic_op_from_ints_both(self):
        """Tests that the correct FermionicOp is built from 1- and 2-body integrals."""
        expected_num_of_terms_ferm_op = 184
        expected_fermionic_op_path = self.get_resource_path(
            "H2_631g_ferm_op_two_ints",
            "problems/second_quantization/" "electronic/resources",
        )
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_631g.hdf5", "transformers")
        )
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            q_molecule.one_body_integrals, q_molecule.two_body_integrals
        )

        with self.subTest("Check type of fermionic operator"):
            assert isinstance(fermionic_op, FermionicOp)
        with self.subTest("Check expected number of terms in a fermionic operator."):
            assert len(fermionic_op) == expected_num_of_terms_ferm_op
        with self.subTest("Check expected content of a fermionic operator."):
            assert all(
                s[0] == t[0] and np.isclose(s[1], t[1])
                for s, t in zip(fermionic_op.to_list(), expected_fermionic_op)
            )

    def test_build_fermionic_op_from_ints_one(self):
        """Tests that the correct FermionicOp is built from 1-body integrals."""
        expected_num_of_terms_ferm_op = 16
        expected_fermionic_op_path = self.get_resource_path(
            "H2_631g_ferm_op_one_int",
            "problems/second_quantization/" "electronic/resources",
        )
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)

        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_631g.hdf5", "transformers")
        )
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            q_molecule.one_body_integrals
        )

        with self.subTest("Check type of fermionic operator"):
            assert isinstance(fermionic_op, FermionicOp)
        with self.subTest("Check expected number of terms in a fermionic operator."):
            assert len(fermionic_op) == expected_num_of_terms_ferm_op
        with self.subTest("Check expected content of a fermionic operator."):
            assert all(
                s[0] == t[0] and np.isclose(s[1], t[1])
                for s, t in zip(fermionic_op.to_list(), expected_fermionic_op)
            )
