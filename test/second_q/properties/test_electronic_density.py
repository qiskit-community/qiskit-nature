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

    @data(True, False)
    def test_from_orbital_occupation(self, include_rdm2: bool):
        """Test from_orbital_occupation."""
        rdm2_ba = np.zeros((2, 2, 2, 2))
        rdm2_ba[0, 0, 0, 0] = 1.0
        expected = ElectronicIntegrals.from_raw_integrals(
            np.diag([1, 0]),
            np.zeros((2, 2, 2, 2)) if include_rdm2 else None,
            np.diag([1, 0]),
            np.zeros((2, 2, 2, 2)) if include_rdm2 else None,
            rdm2_ba if include_rdm2 else None,
        )
        density = ElectronicDensity.from_orbital_occupation(
            [1, 0], [1, 0], include_rdm2=include_rdm2
        )
        self.assertTrue(density.equiv(expected))

    @data(True, False)
    def test_second_q_ops(self, include_rdm2: bool):
        """Test second_q_ops."""
        density = ElectronicDensity.from_orbital_occupation(
            [1, 0], [1, 0], include_rdm2=include_rdm2
        )
        aux_ops = density.second_q_ops()

        with self.subTest("operator keys"):
            expected_keys = set()
            expected_keys.update({f"RDM{index}" for index in product([0, 1], repeat=2)})
            expected_keys.update({f"RDM{index}" for index in product([2, 3], repeat=2)})
            if include_rdm2:
                expected_keys.update({f"RDM{index}" for index in product([0, 1], repeat=4)})
                expected_keys.update({f"RDM{index}" for index in product([2, 3], repeat=4)})
                expected_keys.update(
                    {
                        f"RDM{index[0][0], index[1][0], index[1][1], index[0][1]}"
                        for index in product(product([0, 1], repeat=2), product([2, 3], repeat=2))
                    }
                )
            self.assertEqual(expected_keys, aux_ops.keys())

        with self.subTest("operator contents"):
            all_terms = {}
            for op in aux_ops.values():
                all_terms.update(op.items())

            expected_terms = {}
            expected_terms.update(
                {f"+_{index[0]} -_{index[1]}": 1.0 for index in product([0, 1], repeat=2)}
            )
            expected_terms.update(
                {f"+_{index[0]} -_{index[1]}": 1.0 for index in product([2, 3], repeat=2)}
            )
            if include_rdm2:
                expected_terms.update(
                    {
                        f"+_{index[0]} +_{index[1]} -_{index[2]} -_{index[3]}": 1.0
                        for index in product([0, 1], repeat=4)
                    }
                )
                expected_terms.update(
                    {
                        f"+_{index[0]} +_{index[1]} -_{index[2]} -_{index[3]}": 1.0
                        for index in product([2, 3], repeat=4)
                    }
                )
                expected_terms.update(
                    {
                        f"+_{index[0][0]} +_{index[1][0]} -_{index[1][1]} -_{index[0][1]}": 1.0
                        for index in product(product([0, 1], repeat=2), product([2, 3], repeat=2))
                    }
                )

            self.assertEqual(expected_terms, all_terms)

    @data(True, False)
    def test_interpret(self, include_rdm2: bool):
        """Test interpret."""
        dummy_result = ElectronicStructureResult()
        aux_values = {}
        aux_values.update({f"RDM{index}": 1.0 for index in product([0, 1], repeat=2)})
        aux_values.update({f"RDM{index}": 1.0 for index in product([2, 3], repeat=2)})
        if include_rdm2:
            aux_values.update({f"RDM{index}": 1.0 for index in product([0, 1], repeat=4)})
            aux_values.update({f"RDM{index}": 1.0 for index in product([2, 3], repeat=4)})
            aux_values.update(
                {
                    f"RDM{index[0][0], index[1][0], index[1][1], index[0][1]}": 1.0
                    for index in product(product([0, 1], repeat=2), product([2, 3], repeat=2))
                }
            )
        dummy_result.aux_operators_evaluated = [aux_values]

        density = ElectronicDensity.from_orbital_occupation(
            [1, 0], [1, 0], include_rdm2=include_rdm2
        )
        density.interpret(dummy_result)

        expected = ElectronicIntegrals.from_raw_integrals(
            np.ones((2, 2)),
            np.ones((2, 2, 2, 2)) if include_rdm2 else None,
            np.ones((2, 2)),
            np.ones((2, 2, 2, 2)) if include_rdm2 else None,
            np.ones((2, 2, 2, 2)) if include_rdm2 else None,
        )

        self.assertTrue(dummy_result.electronic_density.equiv(expected))

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @data(MethodType.RHF, MethodType.UHF)
    def test_evaluated_densities(self, method):
        """A validation test against PySCF."""
        driver = PySCFDriver(method=method)
        problem = driver.run()
        electronic_density = ElectronicDensity.from_orbital_occupation(
            problem.orbital_occupations,
            problem.orbital_occupations_b,
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

        norb = problem.num_spatial_orbitals
        nelec = problem.num_particles

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
