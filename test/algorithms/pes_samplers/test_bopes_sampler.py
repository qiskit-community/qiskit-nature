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

"""Tests of BOPES Sampler."""

import unittest
from functools import partial

from test import QiskitNatureTestCase

import numpy as np

import qiskit
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.utils import algorithm_globals, QuantumInstance, optionals
import qiskit_nature.optionals as _optionals

from qiskit_nature.algorithms import (
    GroundStateEigensolver,
    BOPESSampler,
    ExcitedStatesEigensolver,
    NumPyEigensolverFactory,
    QEOM,
)
from qiskit_nature.algorithms.pes_samplers import MorsePotential
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import (
    VQEUCCFactory,
)
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem


class TestBOPES(QiskitNatureTestCase):
    """Tests of BOPES Sampler."""

    def setUp(self) -> None:
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_h2_bopes_sampler(self):
        """Test BOPES Sampler on H2"""
        # Molecule
        dof = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        m = Molecule(
            geometry=[["H", [0.0, 0.0, 1.0]], ["H", [0.0, 0.45, 1.0]]],
            degrees_of_freedom=[dof],
        )

        mapper = ParityMapper()
        converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

        driver = ElectronicStructureMoleculeDriver(
            m, driver_type=ElectronicStructureDriverType.PYSCF
        )
        problem = ElectronicStructureProblem(driver)

        solver = NumPyMinimumEigensolver()
        me_gss = GroundStateEigensolver(converter, solver)

        # BOPES sampler
        sampler = BOPESSampler(me_gss)

        # absolute internuclear distance in Angstrom
        points = [0.7, 1.0, 1.3]
        results = sampler.sample(problem, points)

        points_run = results.points
        energies = results.energies

        np.testing.assert_array_almost_equal(points_run, [0.7, 1.0, 1.3])
        np.testing.assert_array_almost_equal(
            energies, [-1.13618945, -1.10115033, -1.03518627], decimal=2
        )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_potential_interface(self):
        """Tests potential interface."""
        stretch = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        # H-H molecule near equilibrium geometry
        m = Molecule(
            geometry=[
                ["H", [0.0, 0.0, 0.0]],
                ["H", [1.0, 0.0, 0.0]],
            ],
            degrees_of_freedom=[stretch],
            masses=[1.6735328e-27, 1.6735328e-27],
        )

        mapper = ParityMapper()
        converter = QubitConverter(mapper=mapper)

        driver = ElectronicStructureMoleculeDriver(
            m, driver_type=ElectronicStructureDriverType.PYSCF
        )
        problem = ElectronicStructureProblem(driver)

        solver = NumPyMinimumEigensolver()

        me_gss = GroundStateEigensolver(converter, solver)
        # Run BOPESSampler with exact eigensolution
        points = np.arange(0.45, 5.3, 0.3)
        sampler = BOPESSampler(me_gss)

        res = sampler.sample(problem, points)

        # Testing Potential interface
        pot = MorsePotential(m)
        pot.fit(res.points, res.energies)

        np.testing.assert_array_almost_equal([pot.alpha, pot.r_0], [2.235, 0.720], decimal=3)
        np.testing.assert_array_almost_equal([pot.d_e, pot.m_shift], [0.2107, -1.1419], decimal=3)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_vqe_bootstrap(self):
        """Test with VQE and bootstrapping."""
        qubit_converter = QubitConverter(JordanWignerMapper())
        quantum_instance = QuantumInstance(
            backend=qiskit.Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        solver = VQE(quantum_instance=quantum_instance)

        vqe_gse = GroundStateEigensolver(qubit_converter, solver)

        distance1 = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        mol = Molecule(
            geometry=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.6])],
            degrees_of_freedom=[distance1],
        )

        driver = ElectronicStructureMoleculeDriver(
            mol, driver_type=ElectronicStructureDriverType.PYSCF
        )
        es_problem = ElectronicStructureProblem(driver)
        points = list(np.linspace(0.6, 0.8, 4))
        bopes = BOPESSampler(vqe_gse, bootstrap=True, num_bootstrap=None, extrapolator=None)
        result = bopes.sample(es_problem, points)
        ref_points = [0.6, 0.6666666666666666, 0.7333333333333334, 0.8]
        ref_energies = [
            -1.1162738,
            -1.1326904,
            -1.1372876,
            -1.1341292,
        ]
        np.testing.assert_almost_equal(result.points, ref_points)
        np.testing.assert_almost_equal(result.energies, ref_energies)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_h2_bopes_sampler_with_factory(self):
        """Test BOPES Sampler with Factory"""
        quantum_instance = QuantumInstance(
            backend=qiskit.Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        # Molecule
        distance1 = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        molecule = Molecule(
            geometry=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.6])],
            degrees_of_freedom=[distance1],
        )

        driver = ElectronicStructureMoleculeDriver(
            molecule, driver_type=ElectronicStructureDriverType.PYSCF
        )
        problem = ElectronicStructureProblem(driver)
        converter = QubitConverter(ParityMapper())
        solver = GroundStateEigensolver(converter, VQEUCCFactory(quantum_instance=quantum_instance))
        sampler = BOPESSampler(solver, bootstrap=True, num_bootstrap=None, extrapolator=None)

        result = sampler.sample(problem, list(np.linspace(0.6, 0.8, 4)))

        ref_points = [0.6, 0.6666666666666666, 0.7333333333333334, 0.8]
        ref_energies = [
            -1.1162853926251162,
            -1.1327033478688526,
            -1.137302817836066,
            -1.1341458916990401,
        ]
        np.testing.assert_almost_equal(result.points, ref_points, decimal=3)
        np.testing.assert_almost_equal(result.energies, ref_energies, decimal=3)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_h2_bopes_sampler_excited_eigensolver(self):
        """Test BOPES Sampler on H2"""
        # Molecule
        dof = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        m = Molecule(
            geometry=[["H", [0.0, 0.0, 1.0]], ["H", [0.0, 0.45, 1.0]]],
            degrees_of_freedom=[dof],
        )

        mapper = ParityMapper()
        converter = QubitConverter(mapper=mapper)

        driver = ElectronicStructureMoleculeDriver(
            m, driver_type=ElectronicStructureDriverType.PYSCF
        )
        problem = ElectronicStructureProblem(driver)

        # pylint: disable=unused-argument
        def filter_criterion(eigenstate, eigenvalue, aux_values):
            particle_number_filter = np.isclose(aux_values["ParticleNumber"][0], 2.0)
            magnetization_filter = np.isclose(aux_values["Magnetization"][0], 0.0)
            return particle_number_filter and magnetization_filter

        solver = NumPyEigensolverFactory(filter_criterion=filter_criterion)
        np_excited_solver = ExcitedStatesEigensolver(converter, solver)

        # BOPES sampler
        sampler = BOPESSampler(np_excited_solver)

        # absolute internuclear distance in Angstrom
        points = [0.7, 1.0, 1.3]
        results = sampler.sample(problem, points)

        points_run = results.points
        energies = results.energies

        np.testing.assert_array_almost_equal(points_run, [0.7, 1.0, 1.3])
        np.testing.assert_array_almost_equal(
            energies,
            [
                [-1.13618945, -0.47845306, -0.1204519, 0.5833141],
                [-1.10115033, -0.74587179, -0.35229063, 0.03904763],
                [-1.03518627, -0.85523694, -0.42240202, -0.21860355],
            ],
            decimal=2,
        )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_h2_bopes_sampler_qeom(self):
        """Test BOPES Sampler on H2"""
        # Molecule
        dof = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        m = Molecule(
            geometry=[["H", [0.0, 0.0, 1.0]], ["H", [0.0, 0.45, 1.0]]],
            degrees_of_freedom=[dof],
        )
        driver = ElectronicStructureMoleculeDriver(
            m, driver_type=ElectronicStructureDriverType.PYSCF
        )
        problem = ElectronicStructureProblem(driver)

        qubit_converter = QubitConverter(JordanWignerMapper(), z2symmetry_reduction=None)
        quantum_instance = QuantumInstance(
            backend=qiskit.Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        solver = VQE(quantum_instance=quantum_instance)
        me_gsc = GroundStateEigensolver(qubit_converter, solver)
        qeom_solver = QEOM(me_gsc, "sd")

        # BOPES sampler
        sampler = BOPESSampler(qeom_solver)

        # absolute internuclear distance in Angstrom
        points = [0.7, 1.0, 1.3]
        results = sampler.sample(problem, points)

        points_run = results.points
        energies = results.energies

        np.testing.assert_array_almost_equal(points_run, [0.7, 1.0, 1.3])
        np.testing.assert_array_almost_equal(
            energies,
            [
                [-1.13618945, -0.47845306, -0.1204519, 0.5833141],
                [-1.10115033, -0.74587179, -0.35229063, 0.03904763],
                [-1.03518627, -0.85523694, -0.42240202, -0.21860355],
            ],
            decimal=2,
        )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_h2_bopes_sampler_auxiliaries(self):
        """Test BOPES Sampler on H2"""
        # Molecule
        dof = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        m = Molecule(
            geometry=[["H", [0.0, 0.0, 1.0]], ["H", [0.0, 0.45, 1.0]]],
            degrees_of_freedom=[dof],
        )
        driver = ElectronicStructureMoleculeDriver(
            m, driver_type=ElectronicStructureDriverType.PYSCF
        )
        problem = ElectronicStructureProblem(driver)

        qubit_converter = QubitConverter(JordanWignerMapper(), z2symmetry_reduction=None)
        quantum_instance = QuantumInstance(
            backend=qiskit.Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        solver = VQE(quantum_instance=quantum_instance)
        me_gsc = GroundStateEigensolver(qubit_converter, solver)

        # BOPES sampler
        aux = {"PN": problem.second_q_ops()["ParticleNumber"]}
        sampler = BOPESSampler(me_gsc)

        # absolute internuclear distance in Angstrom
        points = [0.7, 1.0, 1.3]
        results = sampler.sample(problem, points, aux_operators=aux)

        points_run = results.points
        particle_number = []
        for results_point in list(results.raw_results.values()):
            particle_number.append(results_point.raw_result.aux_operator_eigenvalues["PN"][0])

        np.testing.assert_array_almost_equal(points_run, [0.7, 1.0, 1.3])
        np.testing.assert_array_almost_equal(
            particle_number,
            [2, 2, 2],
            decimal=2,
        )


if __name__ == "__main__":
    unittest.main()
