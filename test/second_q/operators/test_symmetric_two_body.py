# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the symmetric 2-body electronic integral utilities."""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit_algorithms import NumPyMinimumEigensolver

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import (
    get_ao_to_mo_from_qcschema,
    qcschema_to_problem,
)
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp, PolynomialTensor
from qiskit_nature.second_q.operators.symmetric_two_body import (
    S1Integrals,
    S4Integrals,
    S8Integrals,
    unfold,
    unfold_s4_to_s1,
    unfold_s8_to_s1,
    unfold_s8_to_s4,
    fold,
    fold_s1_to_s4,
    fold_s1_to_s8,
    fold_s4_to_s8,
)
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem


@ddt
class TestSymmetricIntegrals(QiskitNatureTestCase):
    """Tests the symmetric 2-body electronic integral utility classes."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "PySCF not available.")
    def setUp(self):
        super().setUp()

        # pylint: disable=import-error
        from pyscf import gto

        self.mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")

        self.eri1 = self.mol.intor("int2e", aosym=1)
        self.eri4 = self.mol.intor("int2e", aosym=4)
        self.eri8 = self.mol.intor("int2e", aosym=8)

        self.ints1 = S1Integrals(self.eri1)
        self.ints4 = S4Integrals(self.eri4)
        self.ints8 = S8Integrals(self.eri8)

        self.reference = FermionicOp.from_polynomial_tensor(
            PolynomialTensor({"++--": to_physicist_ordering(self.eri1)})
        )

    @data("ints1", "ints4", "ints8")
    def test_unfold(self, ints_name: str):
        """Test the ``unfold`` method."""
        ints = getattr(self, ints_name)
        unfolded = unfold(ints)
        if not unfolded.is_dense():
            unfolded = unfolded.to_dense()
        np.testing.assert_allclose(self.eri1, unfolded)

    def test_unfold_s4_to_s1(self):
        """Test the ``unfold_s4_to_s1`` method."""
        unfolded = unfold_s4_to_s1(self.ints4)
        if not unfolded.is_dense():
            unfolded = unfolded.to_dense()
        np.testing.assert_allclose(self.eri1, unfolded)

    def test_unfold_s8_to_s1(self):
        """Test the ``unfold_s8_to_s1`` method."""
        unfolded = unfold_s8_to_s1(self.ints8)
        if not unfolded.is_dense():
            unfolded = unfolded.to_dense()
        np.testing.assert_allclose(self.eri1, unfolded)

    def test_unfold_s8_to_s4(self):
        """Test the ``unfold_s8_to_s4`` method."""
        np.testing.assert_allclose(self.eri4, unfold_s8_to_s4(self.ints8))

    @data("ints1", "ints4", "ints8")
    def test_fold(self, ints_name: str):
        """Test the ``fold`` method."""
        ints = getattr(self, ints_name)
        np.testing.assert_allclose(self.eri8, fold(ints))

    def test_fold_s1_to_s4(self):
        """Test the ``fold_s1_to_s4`` method."""
        np.testing.assert_allclose(self.eri4, fold_s1_to_s4(self.ints1))

    def test_fold_s1_to_s4_failure(self):
        """Test the ``fold_s1_to_s4`` method failure."""
        with self.assertRaises(ValueError):
            fold_s1_to_s4(S1Integrals(np.arange(16).reshape((2, 2, 2, 2))))

    def test_fold_s1_to_s8(self):
        """Test the ``fold_s1_to_s8`` method."""
        np.testing.assert_allclose(self.eri8, fold_s1_to_s8(self.ints1))

    def test_fold_s1_to_s8_failure(self):
        """Test the ``fold_s1_to_s8`` method failure."""
        with self.assertRaises(ValueError):
            fold_s1_to_s8(S1Integrals(np.arange(16).reshape((2, 2, 2, 2))))

    def test_fold_s4_to_s8(self):
        """Test the ``fold_s4_to_s8`` method."""
        np.testing.assert_allclose(self.eri8, fold_s4_to_s8(self.ints4))

    def test_fold_s4_to_s8_failure(self):
        """Test the ``fold_s4_to_s8`` method failure."""
        with self.assertRaises(ValueError):
            fold_s4_to_s8(S4Integrals(np.arange(9).reshape((3, 3))))

    def test_s4_getitem(self):
        """Test item access in the ``S4Integrals``."""
        for index in np.ndindex(*self.ints1.shape):
            self.assertAlmostEqual(self.ints1[index], self.ints4[index])

    def test_s8_getitem(self):
        """Test item access in the ``S8Integrals``."""
        for index in np.ndindex(*self.ints1.shape):
            self.assertAlmostEqual(self.ints1[index], self.ints8[index])

    def test_s1_fermionic_op(self):
        """Test the FermionicOp generated from a S1Integrals instance."""
        self.assertTrue(
            self.reference.equiv(
                FermionicOp.from_polynomial_tensor(PolynomialTensor({"++--": self.ints1}))
            )
        )

    def test_s4_fermionic_op(self):
        """Test the FermionicOp generated from a S4Integrals instance."""
        self.assertTrue(
            self.reference.equiv(
                FermionicOp.from_polynomial_tensor(PolynomialTensor({"++--": self.ints4}))
            )
        )

    @unittest.skipIf(not _optionals.HAS_SPARSE, "sparse library not available.")
    def test_s4_fermionic_op_sparse(self):
        """Test the FermionicOp generated from a sparse S4Integrals instance."""
        self.assertTrue(
            self.reference.equiv(
                FermionicOp.from_polynomial_tensor(
                    PolynomialTensor({"++--": self.ints4.to_sparse()})
                )
            )
        )

    def test_s8_fermionic_op(self):
        """Test the FermionicOp generated from a S8Integrals instance."""
        self.assertTrue(
            self.reference.equiv(
                FermionicOp.from_polynomial_tensor(PolynomialTensor({"++--": self.ints8}))
            )
        )

    @unittest.skipIf(not _optionals.HAS_SPARSE, "sparse library not available.")
    def test_s8_fermionic_op_sparse(self):
        """Test the FermionicOp generated from a sparse S8Integrals instance."""
        self.assertTrue(
            self.reference.equiv(
                FermionicOp.from_polynomial_tensor(
                    PolynomialTensor({"++--": self.ints8.to_sparse()})
                )
            )
        )

    def test_s4_integration(self):
        """Test integration of the S4Integrals."""
        # pylint: disable=import-error
        from pyscf import ao2mo

        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolver())
        driver = PySCFDriver()
        problem = driver.run()
        expected = algo.solve(problem).computed_energies[0]

        s4_hamil = ElectronicEnergy(
            ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": np.dot(
                            np.dot(driver._calc.mo_coeff.T, driver._calc.get_hcore()),
                            driver._calc.mo_coeff,
                        ),
                        "++--": S4Integrals(
                            ao2mo.full(driver._mol, driver._calc.mo_coeff, aosym=4)
                        ),
                    },
                    validate=False,
                )
            )
        )
        s4_problem = ElectronicStructureProblem(s4_hamil)
        result = algo.solve(s4_problem)
        with self.subTest("computed energy"):
            self.assertAlmostEqual(expected, result.computed_energies[0])
        with self.subTest("generated FermionicOp"):
            self.assertTrue(
                problem.hamiltonian.second_q_op().equiv(s4_problem.hamiltonian.second_q_op())
            )

    def test_s4_integration_uhf(self):
        """Test integration of the S4Integrals with UHF."""
        # pylint: disable=import-error
        from pyscf import ao2mo

        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolver())
        driver = PySCFDriver(method=MethodType.UHF)
        problem = driver.run()
        expected = algo.solve(problem).computed_energies[0]

        s4_hamil = ElectronicEnergy(
            ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": np.dot(
                            np.dot(driver._calc.mo_coeff[0].T, driver._calc.get_hcore()),
                            driver._calc.mo_coeff[0],
                        ),
                        "++--": S4Integrals(
                            ao2mo.full(driver._mol, driver._calc.mo_coeff[0], aosym=4)
                        ),
                    },
                    validate=False,
                ),
                PolynomialTensor(
                    {
                        "+-": np.dot(
                            np.dot(driver._calc.mo_coeff[1].T, driver._calc.get_hcore()),
                            driver._calc.mo_coeff[1],
                        ),
                        "++--": S4Integrals(
                            ao2mo.full(driver._mol, driver._calc.mo_coeff[1], aosym=4)
                        ),
                    },
                    validate=False,
                ),
                PolynomialTensor(
                    {
                        "++--": S4Integrals(
                            ao2mo.general(
                                driver._mol,
                                [
                                    driver._calc.mo_coeff[1],
                                    driver._calc.mo_coeff[1],
                                    driver._calc.mo_coeff[0],
                                    driver._calc.mo_coeff[0],
                                ],
                                aosym=4,
                            )
                        ),
                    },
                    validate=False,
                ),
            )
        )
        s4_problem = ElectronicStructureProblem(s4_hamil)
        result = algo.solve(s4_problem)
        with self.subTest("computed energy"):
            self.assertAlmostEqual(expected, result.computed_energies[0])
        with self.subTest("generated FermionicOp"):
            self.assertTrue(
                problem.hamiltonian.second_q_op().equiv(s4_problem.hamiltonian.second_q_op())
            )

    def test_s8_integration(self):
        """Test integration of the S8Integrals."""
        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolver())
        driver = PySCFDriver()
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        trafo = get_ao_to_mo_from_qcschema(qcschema)
        problem = qcschema_to_problem(qcschema, basis=ElectronicBasis.MO)
        expected = algo.solve(problem).computed_energies[0]

        s8_hamil = ElectronicEnergy(
            ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": driver._calc.get_hcore(),
                        "++--": S8Integrals(driver._mol.intor("int2e", aosym=8)),
                    },
                    validate=False,
                )
            )
        )
        ao_problem = ElectronicStructureProblem(s8_hamil)
        ao_problem.basis = ElectronicBasis.AO
        s8_mo_problem = trafo.transform(ao_problem)
        result = algo.solve(s8_mo_problem)
        with self.subTest("computed energy"):
            self.assertAlmostEqual(expected, result.computed_energies[0])
        with self.subTest("generated FermionicOp"):
            self.assertTrue(
                problem.hamiltonian.second_q_op().equiv(s8_mo_problem.hamiltonian.second_q_op())
            )

    def test_s8_integration_uhf(self):
        """Test integration of the S8Integrals with UHF."""
        # pylint: disable=import-error
        from pyscf import ao2mo

        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolver())
        driver = PySCFDriver(method=MethodType.UHF)
        problem = driver.run()
        expected = algo.solve(problem).computed_energies[0]

        s8_hamil = ElectronicEnergy(
            ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": np.dot(
                            np.dot(driver._calc.mo_coeff[0].T, driver._calc.get_hcore()),
                            driver._calc.mo_coeff[0],
                        ),
                        "++--": fold_s4_to_s8(
                            ao2mo.full(driver._mol, driver._calc.mo_coeff[0], aosym=4)
                        ),
                    },
                    validate=False,
                ),
                PolynomialTensor(
                    {
                        "+-": np.dot(
                            np.dot(driver._calc.mo_coeff[1].T, driver._calc.get_hcore()),
                            driver._calc.mo_coeff[1],
                        ),
                        "++--": fold_s4_to_s8(
                            ao2mo.full(driver._mol, driver._calc.mo_coeff[1], aosym=4)
                        ),
                    },
                    validate=False,
                ),
                PolynomialTensor(
                    {
                        "++--": fold_s4_to_s8(
                            ao2mo.general(
                                driver._mol,
                                [
                                    driver._calc.mo_coeff[1],
                                    driver._calc.mo_coeff[1],
                                    driver._calc.mo_coeff[0],
                                    driver._calc.mo_coeff[0],
                                ],
                                aosym=4,
                            )
                        ),
                    },
                    validate=False,
                ),
            )
        )
        s8_problem = ElectronicStructureProblem(s8_hamil)
        result = algo.solve(s8_problem)
        with self.subTest("computed energy"):
            self.assertAlmostEqual(expected, result.computed_energies[0])
        with self.subTest("generated FermionicOp"):
            self.assertTrue(
                problem.hamiltonian.second_q_op().equiv(s8_problem.hamiltonian.second_q_op())
            )


if __name__ == "__main__":
    unittest.main()
