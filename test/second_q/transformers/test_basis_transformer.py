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

"""Tests for the BasisTransformer."""

import unittest
from typing import cast

from test import QiskitNatureTestCase

import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.formats.qcschema_translator import get_ao_to_mo_from_qcschema
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicBasis


class TestBasisTransformer(QiskitNatureTestCase):
    """BasisTransformer tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_restricted_spin(self):
        """A simple restricted-spin test case."""
        driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", method=MethodType.RKS)
        driver.run_pyscf()
        mo_coeff = driver._calc.mo_coeff

        problem_ao = driver.to_problem(basis=ElectronicBasis.AO)
        qcschema = driver.to_qcschema()

        trafo = get_ao_to_mo_from_qcschema(qcschema)

        with self.subTest("bases"):
            self.assertEqual(trafo.initial_basis, ElectronicBasis.AO)
            self.assertEqual(trafo.final_basis, ElectronicBasis.MO)

        with self.subTest("alpha-spin coefficients"):
            np.testing.assert_array_almost_equal(trafo.coefficients.alpha["+-"], mo_coeff)

        with self.subTest("empty beta-spin coefficients"):
            self.assertTrue(trafo.coefficients.beta.is_empty())

        with self.subTest("empty beta-alpha-spin coefficients"):
            self.assertTrue(trafo.coefficients.beta_alpha.is_empty())

        problem_mo = trafo.transform(problem_ao)
        transformed_integrals = cast(ElectronicEnergy, problem_mo.hamiltonian).electronic_integrals

        with self.subTest("one-body alpha-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.alpha["+-"],
                np.dot(np.dot(mo_coeff.T, driver._calc.get_hcore()), mo_coeff),
            )

        with self.subTest("two-body alpha-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.alpha["++--"],
                np.einsum(
                    "pqrs,pi,qj,rk,sl->iklj",
                    driver._mol.intor("int2e", aosym=1),
                    *(mo_coeff,) * 4,
                    optimize=True,
                ),
            )

        with self.subTest("beta-spin is empty"):
            self.assertTrue(transformed_integrals.beta.is_empty())

        with self.subTest("beta-alpha-spin is empty"):
            self.assertTrue(transformed_integrals.beta_alpha.is_empty())

        with self.subTest("attributes"):
            self.assertEqual(problem_ao.molecule, problem_mo.molecule)
            self.assertEqual(problem_ao.reference_energy, problem_mo.reference_energy)
            self.assertEqual(problem_ao.num_particles, problem_mo.num_particles)
            self.assertEqual(problem_ao.num_spatial_orbitals, problem_mo.num_spatial_orbitals)
            self.assertIsNone(problem_mo.orbital_energies)
            self.assertIsNone(problem_mo.orbital_energies_b)
            # orbital_occupations are not tested since in the MO basis they are auto-filled

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_unrestricted_spin(self):
        """A simple unrestricted-spin test case."""
        driver = PySCFDriver(atom="O 0 0 0; H 0 0 0.9697", spin=1, method=MethodType.UKS)
        driver.run_pyscf()
        mo_coeff, mo_coeff_b = driver._calc.mo_coeff

        problem_ao = driver.to_problem(basis=ElectronicBasis.AO)
        qcschema = driver.to_qcschema()

        trafo = get_ao_to_mo_from_qcschema(qcschema)

        with self.subTest("bases"):
            self.assertEqual(trafo.initial_basis, ElectronicBasis.AO)
            self.assertEqual(trafo.final_basis, ElectronicBasis.MO)

        with self.subTest("alpha-spin coefficients"):
            np.testing.assert_array_almost_equal(trafo.coefficients.alpha["+-"], mo_coeff)

        with self.subTest("beta-spin coefficients"):
            np.testing.assert_array_almost_equal(trafo.coefficients.beta["+-"], mo_coeff_b)

        with self.subTest("empty beta-alpha-spin coefficients"):
            self.assertTrue(trafo.coefficients.beta_alpha.is_empty())

        problem_mo = trafo.transform(problem_ao)
        transformed_integrals = cast(ElectronicEnergy, problem_mo.hamiltonian).electronic_integrals

        with self.subTest("one-body alpha-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.alpha["+-"],
                np.dot(np.dot(mo_coeff.T, driver._calc.get_hcore()), mo_coeff),
            )

        with self.subTest("two-body alpha-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.alpha["++--"],
                np.einsum(
                    "pqrs,pi,qj,rk,sl->iklj",
                    driver._mol.intor("int2e", aosym=1),
                    *(mo_coeff,) * 4,
                    optimize=True,
                ),
            )

        with self.subTest("one-body beta-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.beta["+-"],
                np.dot(np.dot(mo_coeff_b.T, driver._calc.get_hcore()), mo_coeff_b),
            )

        with self.subTest("two-body beta-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.beta["++--"],
                np.einsum(
                    "pqrs,pi,qj,rk,sl->iklj",
                    driver._mol.intor("int2e", aosym=1),
                    *(mo_coeff_b,) * 4,
                    optimize=True,
                ),
            )

        with self.subTest("two-body beta-alpha-spin"):
            np.testing.assert_array_almost_equal(
                transformed_integrals.beta_alpha["++--"],
                np.einsum(
                    "pqrs,pi,qj,rk,sl->iklj",
                    driver._mol.intor("int2e", aosym=1),
                    *(mo_coeff_b,) * 2,
                    *(mo_coeff,) * 2,
                    optimize=True,
                ),
            )

        with self.subTest("attributes"):
            self.assertEqual(problem_ao.molecule, problem_mo.molecule)
            self.assertEqual(problem_ao.reference_energy, problem_mo.reference_energy)
            self.assertEqual(problem_ao.num_particles, problem_mo.num_particles)
            self.assertEqual(problem_ao.num_spatial_orbitals, problem_mo.num_spatial_orbitals)
            self.assertIsNone(problem_mo.orbital_energies)
            self.assertIsNone(problem_mo.orbital_energies_b)
            # orbital_occupations are not tested since in the MO basis they are auto-filled


if __name__ == "__main__":
    unittest.main()
