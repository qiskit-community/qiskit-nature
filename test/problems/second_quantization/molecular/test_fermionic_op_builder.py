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
from qiskit_nature.operators import FermionicOp
from qiskit_nature.problems.second_quantization.molecular import fermionic_op_builder
from qiskit_nature.drivers import HDF5Driver


class TestFermionicOperatorBuilder(QiskitNatureTestCase):
    """Tests Fermionic Operator builder."""

    def test_build_fermionic_op(self):
        """Tests that the correct FermionicOp is built from QMolecule."""
        expected_fermionic_op_path = self.get_resource_path('H2_631g_ferm_op_two_ints',
                                                            'problems/second_quantization/'
                                                            'molecular')
        expected_fermionic_op = self._read_expected_file(expected_fermionic_op_path)
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_fermionic_op(q_molecule)
        assert isinstance(fermionic_op, FermionicOp)
        assert len(fermionic_op) == 184
        assert fermionic_op.to_list() == expected_fermionic_op

    def test_build_fermionic_op_from_ints_both(self):
        """Tests that the correct FermionicOp is built from 1- and 2-body integrals."""
        expected_fermionic_op_path = self.get_resource_path('H2_631g_ferm_op_two_ints',
                                                            'problems/second_quantization/'
                                                            'molecular')
        expected_fermionic_op = self._read_expected_file(expected_fermionic_op_path)
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            q_molecule.one_body_integrals, q_molecule.two_body_integrals)
        assert isinstance(fermionic_op, FermionicOp)
        assert len(fermionic_op) == 184
        assert fermionic_op.to_list() == expected_fermionic_op

    def test_build_fermionic_op_from_ints_one(self):
        """Tests that the correct FermionicOp is built from 1-body integrals."""
        expected_fermionic_op_path = self.get_resource_path('H2_631g_ferm_op_one_int',
                                                            'problems/second_quantization/'
                                                            'molecular')
        expected_fermionic_op = self._read_expected_file(expected_fermionic_op_path)

        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            q_molecule.one_body_integrals)
        assert isinstance(fermionic_op, FermionicOp)
        assert len(fermionic_op) == 16
        assert fermionic_op.to_list() == expected_fermionic_op

    @staticmethod
    def _read_expected_file(path):
        types = str, float
        with open(path, 'r') as file:
            expected_fermionic_op = [tuple(t(e) for t, e in zip(types, line.split()))
                                     for line in file]
        return expected_fermionic_op
