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

"""Test AngularMomentum Property"""

import unittest
import json
from test.second_q.properties.property_test import PropertyTest

import numpy as np

from qiskit.primitives import Estimator
from qiskit_algorithms.observables_evaluator import estimate_observables
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.properties import AngularMomentum
from qiskit_nature.second_q.operators import FermionicOp


class TestAngularMomentum(PropertyTest):
    """Test AngularMomentum Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        num_spatial_orbitals = 4
        self.prop = AngularMomentum(num_spatial_orbitals)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        op = self.prop.second_q_ops()["AngularMomentum"]
        with open(
            self.get_resource_path("angular_momentum_op.json", "second_q/properties/resources"),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
            expected_op = FermionicOp(expected, num_spin_orbitals=8)

        self.assertEqual(op.normal_order(), expected_op.normal_order())

    def test_with_overlap(self):
        """Test that the overlap is taken into account.

        The idea of this test is simple. I found the MO coefficients of H2 @ 0.735 Angstrom using a
        RHF calculation (``mo_coeff`` in the code below). Being restricted-spin coefficients for a
        singlet-spin solution, these alone would give a total angular momentum of 0. However, we can
        take the matrix and rotate it by some angle to form a spin contaminated pair of alpha- and
        beta-spin coefficients. These should result in a non-zero angular momentum.

        Below is the code to reproduce the RHF MO coefficients and overlap matrix with PySCF:

        .. code:: python

           from pyscf import gto, scf

           mol = gto.M(atom="H 0.0 0.0 0.0; H 0.0 0.0 0.735;", basis="sto-3g")

           hf = scf.RHF(mol)
           hf.run()

           norb = mol.nao
           nelec = mol.nelec
           mo_coeff = hf.mo_coeff
           ovlp = hf.get_ovlp()

        """
        norb = 2
        nelec = (1, 1)

        mo_coeff = np.asarray([[0.54830202, 1.21832731], [0.54830202, -1.21832731]])
        ovlp = np.asarray([[1.0, 0.66314574], [0.66314574, 1.0]])

        # first, we ensure that our restricted-spin MO coefficients and overlap match
        self.assertTrue(np.allclose(mo_coeff.T @ ovlp @ mo_coeff, np.eye(norb)))

        theta = 33
        rot = np.asarray(
            [
                [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))],
            ],
        )
        mo_coeff_rot = mo_coeff @ rot

        ovlpab = mo_coeff.T @ ovlp @ mo_coeff_rot

        ang_mom = AngularMomentum(norb, ovlpab).second_q_ops()

        mapper = ParityMapper(nelec)
        qubit_op = mapper.map(ang_mom)

        hf_state = HartreeFock(norb, nelec, mapper)

        result = estimate_observables(Estimator(), hf_state, qubit_op)
        self.assertAlmostEqual(result["AngularMomentum"][0], 0.29663167846210015)


if __name__ == "__main__":
    unittest.main()
