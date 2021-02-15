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
from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import PySCFDriver, UnitsType


# TODO add more detailed tests
class TestFermionicOperatorBuilder(QiskitNatureTestCase):
    """Tests Fermionic Operator builder."""

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.595',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
            self.q_molecule = self.driver.run()

        except QiskitNatureError:
            self.skipTest('PYSCF driver does not appear to be installed')

    def test_build_fermionic_op(self):
        """Tests that the FermionicOp is built from QMolecule."""

        fermionic_op = fermionic_op_builder.build_fermionic_op(self.q_molecule)
        assert isinstance(fermionic_op, FermionicOp)

    def test_build_fermionic_op_from_ints_both(self):
        """Tests that the FermionicOp is built from 1- and 2-body integrals."""

        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            self.q_molecule.one_body_integrals, self.q_molecule.two_body_integrals)
        assert isinstance(fermionic_op, FermionicOp)

    def test_build_fermionic_op_from_ints_one(self):
        """Tests that the FermionicOp is built from 1-body integrals."""

        fermionic_op = fermionic_op_builder.build_ferm_op_from_ints(
            self.q_molecule.one_body_integrals)
        assert isinstance(fermionic_op, FermionicOp)
