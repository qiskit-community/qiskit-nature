# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Numerical qEOM excited states calculation """

import unittest
from test import QiskitNatureTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver, NumPyEigensolver

from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.second_quantization.operators.fermionic import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.algorithms import (
    GroundStateEigensolver,
    VQEUCCFactory,
    NumPyEigensolverFactory,
    ExcitedStatesEigensolver,
    QEOM,
)
import qiskit_nature.optionals as _optionals


class TestNumericalQEOMESCCalculation(QiskitNatureTestCase):
    """Test Numerical qEOM excited states calculation"""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.75",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )

        self.reference_energies = [
            -1.8427016,
            -1.8427016 + 0.5943372,
            -1.8427016 + 0.95788352,
            -1.8427016 + 1.5969296,
        ]
        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = ElectronicStructureProblem(self.driver)

        solver = NumPyEigensolver()
        self.ref = solver
        self.quantum_instance = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            seed_transpiler=90,
            seed_simulator=12,
        )

    def test_numpy_mes(self):
        """Test NumPyMinimumEigenSolver with QEOM"""
        solver = NumPyMinimumEigensolver()
        gsc = GroundStateEigensolver(self.qubit_converter, solver)
        esc = QEOM(gsc, "sd")
        results = esc.solve(self.electronic_structure_problem)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_energies[idx], energy, places=4)

    def test_vqe_mes_jw(self):
        """Test VQEUCCSDFactory with QEOM + Jordan Wigner mapping"""
        converter = QubitConverter(JordanWignerMapper())
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_jw_auto(self):
        """Test VQEUCCSDFactory with QEOM + Jordan Wigner mapping + auto symmetry"""
        converter = QubitConverter(JordanWignerMapper(), z2symmetry_reduction="auto")
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_parity(self):
        """Test VQEUCCSDFactory with QEOM + Parity mapping"""
        converter = QubitConverter(ParityMapper())
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_parity_2q(self):
        """Test VQEUCCSDFactory with QEOM + Parity mapping + reduction"""
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_parity_auto(self):
        """Test VQEUCCSDFactory with QEOM + Parity mapping + auto symmetry"""
        converter = QubitConverter(ParityMapper(), z2symmetry_reduction="auto")
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_parity_2q_auto(self):
        """Test VQEUCCSDFactory with QEOM + Parity mapping + reduction + auto symmetry"""
        converter = QubitConverter(
            ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto"
        )
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_bk(self):
        """Test VQEUCCSDFactory with QEOM + Bravyi-Kitaev mapping"""
        converter = QubitConverter(BravyiKitaevMapper())
        self._solve_with_vqe_mes(converter)

    def test_vqe_mes_bk_auto(self):
        """Test VQEUCCSDFactory with QEOM + Bravyi-Kitaev mapping + auto symmetry"""
        converter = QubitConverter(BravyiKitaevMapper(), z2symmetry_reduction="auto")
        self._solve_with_vqe_mes(converter)

    def _solve_with_vqe_mes(self, converter: QubitConverter):
        solver = VQEUCCFactory(quantum_instance=self.quantum_instance)
        gsc = GroundStateEigensolver(converter, solver)
        esc = QEOM(gsc, "sd")
        results = esc.solve(self.electronic_structure_problem)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_energies[idx], energy, places=4)

    def test_numpy_factory(self):
        """Test NumPyEigenSolverFactory with ExcitedStatesEigensolver"""

        # pylint: disable=unused-argument
        def filter_criterion(eigenstate, eigenvalue, aux_values):
            return np.isclose(aux_values["ParticleNumber"][0], 2.0)

        solver = NumPyEigensolverFactory(filter_criterion=filter_criterion)
        esc = ExcitedStatesEigensolver(self.qubit_converter, solver)
        results = esc.solve(self.electronic_structure_problem)

        # filter duplicates from list
        computed_energies = [results.computed_energies[0]]
        for comp_energy in results.computed_energies[1:]:
            if not np.isclose(comp_energy, computed_energies[-1]):
                computed_energies.append(comp_energy)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(computed_energies[idx], energy, places=4)


if __name__ == "__main__":
    unittest.main()
