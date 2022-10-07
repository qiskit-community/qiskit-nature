# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test ElectronicDensity Property"""

from __future__ import annotations

import unittest
from itertools import product
from test import QiskitNatureTestCase

import numpy as np
from ddt import ddt, data

import qiskit_nature.optionals as _optionals

from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.operators.tensor_ordering import _chem_to_phys
from qiskit_nature.second_q.problems import ElectronicStructureResult
from qiskit_nature.second_q.properties import ElectronicDensity


@ddt
class TestElectronicDensity(QiskitNatureTestCase):
    """Test ElectronicDensity Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        alpha_occ = [1, 0]
        beta_occ = [1, 0]
        self.density = ElectronicDensity.from_orbital_occupation(alpha_occ, beta_occ)

    def test_from_orbital_occupation(self):
        """Test from_orbital_occupation."""
        rdm2_ba = np.zeros((2, 2, 2, 2))
        rdm2_ba[0, 0, 0, 0] = 1.0
        expected = ElectronicIntegrals.from_raw_integrals(
            np.diag([1, 0]),
            np.zeros((2, 2, 2, 2)),
            np.diag([1, 0]),
            np.zeros((2, 2, 2, 2)),
            rdm2_ba,
        )
        self.assertTrue(self.density.equiv(expected))

    def test_second_q_ops(self):
        """Test second_q_ops."""
        aux_ops = self.density.second_q_ops()

        with self.subTest("operator keys"):
            expected_keys = set()
            expected_keys.update({f"RDM{index}" for index in product(range(4), repeat=2)})
            expected_keys.update({f"RDM{index}" for index in product(range(4), repeat=4)})
            self.assertEqual(expected_keys, aux_ops.keys())

        with self.subTest("opertor contents"):
            all_terms = {}
            for op in aux_ops.values():
                all_terms.update(op.items())

            expected_terms = {}
            expected_terms.update(
                {f"+_{index[0]} -_{index[1]}": 1.0 for index in product(range(4), repeat=2)}
            )
            expected_terms.update(
                {
                    f"+_{index[0]} +_{index[1]} -_{index[2]} -_{index[3]}": 1.0
                    for index in product(range(4), repeat=4)
                }
            )

            self.assertEqual(expected_terms, all_terms)

    def test_interpret(self):
        """Test interpret."""
        dummy_result = ElectronicStructureResult()
        aux_values = {}
        aux_values.update({f"RDM{index}": 1.0 for index in product(range(4), repeat=2)})
        aux_values.update({f"RDM{index}": 1.0 for index in product(range(4), repeat=4)})
        dummy_result.aux_operators_evaluated = [aux_values]
        self.density.interpret(dummy_result)

        expected = ElectronicIntegrals.from_raw_integrals(
            np.ones((2, 2)),
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2, 2, 2)),
            np.ones((2, 2, 2, 2)),
        )

        self.assertTrue(dummy_result.electronic_density.equiv(expected))

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @data(MethodType.RHF, MethodType.UHF)
    def test_evaluated_densities(self, method):
        """A validation test against PySCF."""
        driver = PySCFDriver(method=method)
        problem = driver.run()
        particle_number = problem.properties.particle_number
        electronic_density = ElectronicDensity.from_orbital_occupation(
            particle_number.occupation_alpha,
            particle_number.occupation_beta,
        )
        problem.properties.electronic_density = electronic_density

        algo = GroundStateEigensolver(
            QubitConverter(JordanWignerMapper()),
            NumPyMinimumEigensolverFactory(),
        )

        result = algo.solve(problem)
        spin_density = result.electronic_density

        # pylint: disable=import-outside-toplevel,import-error
        from pyscf.mcscf import CASCI

        norb = particle_number.num_spin_orbitals // 2
        nelec = particle_number.num_particles

        casci = CASCI(driver._calc, norb, nelec)
        _, _, ci_vec, _, _ = casci.kernel()

        with self.subTest("unrestricted spin matrices"):
            rdm1, rdm2 = casci.fcisolver.make_rdm12s(ci_vec, norb, nelec)
            self.assertTrue(np.allclose(rdm1[0], spin_density.alpha["+-"]))
            self.assertTrue(np.allclose(rdm1[0], spin_density.beta["+-"]))
            self.assertTrue(np.allclose(_chem_to_phys(rdm2[0]), spin_density.alpha["++--"]))
            self.assertTrue(np.allclose(_chem_to_phys(rdm2[2]), spin_density.beta["++--"]))
            self.assertTrue(np.allclose(_chem_to_phys(rdm2[1]), spin_density.beta_alpha["++--"]))

        with self.subTest("spin-traced matrices"):
            traced_density = spin_density.trace_spin()
            rdm1, rdm2 = casci.fcisolver.make_rdm12(ci_vec, norb, nelec)
            self.assertTrue(np.allclose(rdm1, traced_density["+-"]))
            self.assertTrue(np.allclose(_chem_to_phys(rdm2), traced_density["++--"]))


if __name__ == "__main__":
    unittest.main()
