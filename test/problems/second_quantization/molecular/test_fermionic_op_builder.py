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


# TODO add more detailed tests
class TestFermionicOperatorBuilder(QiskitNatureTestCase):
    """Tests Fermionic Operator builder."""

    def test_build_fermionic_op(self):
        """Tests that the FermionicOp is built from QMolecule."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_fermionic_op(q_molecule)
        assert isinstance(fermionic_op, FermionicOp)
        assert len(fermionic_op) == 185

    def test_build_fermionic_op_from_ints_both(self):
        """Tests that the FermionicOp is built from 1- and 2-body integrals."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            q_molecule.one_body_integrals, q_molecule.two_body_integrals)
        assert isinstance(fermionic_op, FermionicOp)
        assert len(fermionic_op) == 185

    def test_build_fermionic_op_from_ints_one(self):
        """Tests that the FermionicOp is built from 1-body integrals."""
        driver = HDF5Driver(hdf5_input=self.get_resource_path('H2_631g.hdf5', 'transformers'))
        q_molecule = driver.run()
        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            q_molecule.one_body_integrals)
        assert isinstance(fermionic_op, FermionicOp)
        assert len(fermionic_op) == 17
