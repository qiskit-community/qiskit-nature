# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests Electronic Structure Problem."""
import unittest
from test import QiskitNatureTestCase
from test.second_q.problems.electronic.resources.resource_reader import (
    read_expected_file,
)

import warnings
import numpy as np

from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.drivers import Molecule
from qiskit_nature.second_q.drivers import (
    HDF5Driver,
    PySCFDriver,
    ElectronicStructureMoleculeDriver,
    ElectronicStructureDriverType,
)
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import (
    ActiveSpaceTransformer,
    FreezeCoreTransformer,
)
import qiskit_nature.optionals as _optionals


class TestElectronicStructureProblem(QiskitNatureTestCase):
    """Tests Electronic Structure Problem."""

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 7
        expected_fermionic_op_path = self.get_resource_path(
            "H2_631g_ferm_op_two_ints",
            "second_q/problems/electronic/resources",
        )
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)

        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_631g.hdf5", "second_q/problems/electronic/transformers"
            )
        )
        electronic_structure_problem = ElectronicStructureProblem(driver)

        second_quantized_ops = electronic_structure_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops[electronic_structure_problem.main_property_name]
        second_quantized_ops = list(second_quantized_ops.values())

        with self.subTest("Check that the correct properties are/aren't None"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # new driver used, molecule_data* should be None
                self.assertIsNone(electronic_structure_problem.molecule_data)
                self.assertIsNone(electronic_structure_problem.molecule_data_transformed)
            # converted properties should never be None
            self.assertIsNotNone(electronic_structure_problem.grouped_property)
            self.assertIsNotNone(electronic_structure_problem.grouped_property_transformed)

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops:
                assert isinstance(second_quantized_op, SecondQuantizedOp)
        with self.subTest("Check components of electronic second quantized operator."):
            assert all(
                s[0] == t[0] and np.isclose(s[1], t[1])
                for s, t in zip(expected_fermionic_op, electr_sec_quant_op.to_list())
            )

    def test_second_q_ops_with_active_space(self):
        """Tests that the correct second quantized operator is created if an active space
        transformer is provided."""
        expected_num_of_sec_quant_ops = 7
        expected_fermionic_op_path = self.get_resource_path(
            "H2_631g_ferm_op_active_space",
            "second_q/problems/electronic/resources",
        )
        expected_fermionic_op = read_expected_file(expected_fermionic_op_path)
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_631g.hdf5", "second_q/problems/electronic/transformers"
            )
        )
        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)

        electronic_structure_problem = ElectronicStructureProblem(driver, [trafo])
        second_quantized_ops = electronic_structure_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops[electronic_structure_problem.main_property_name]
        second_quantized_ops = list(second_quantized_ops.values())

        with self.subTest("Check that the correct properties are/aren't None"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # new driver used, molecule_data* should be None
                self.assertIsNone(electronic_structure_problem.molecule_data)
                self.assertIsNone(electronic_structure_problem.molecule_data_transformed)
            # converted properties should never be None
            self.assertIsNotNone(electronic_structure_problem.grouped_property)
            self.assertIsNotNone(electronic_structure_problem.grouped_property_transformed)

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops:
                assert isinstance(second_quantized_op, SecondQuantizedOp)
        with self.subTest("Check components of electronic second quantized operator."):
            assert all(
                s[0] == t[0] and np.isclose(s[1], t[1])
                for s, t in zip(expected_fermionic_op, electr_sec_quant_op.to_list())
            )


class TestElectronicStructureProblemLegacyDrivers(QiskitNatureTestCase):
    """Tests Electronic Structure Problem."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_sector_locator_h2o(self):
        """Test sector locator."""
        driver = PySCFDriver(
            atom="O 0.0000 0.0000 0.1173; H 0.0000 0.07572 -0.4692;H 0.0000 -0.07572 -0.4692",
            basis="sto-3g",
        )
        es_problem = ElectronicStructureProblem(driver)
        qubit_conv = QubitConverter(
            mapper=ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto"
        )
        qubit_conv.convert(
            es_problem.second_q_ops()[es_problem.main_property_name],
            num_particles=es_problem.num_particles,
            sector_locator=es_problem.symmetry_sector_locator,
        )
        self.assertListEqual(qubit_conv.z2symmetries.tapering_values, [1, -1])

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_sector_locator_homonuclear(self):
        """Test sector locator."""
        molecule = Molecule(
            geometry=[("Li", [0.0, 0.0, 0.0]), ("Li", [0.0, 0.0, 2.771])], charge=0, multiplicity=1
        )
        freeze_core_transformer = FreezeCoreTransformer(True)
        driver = ElectronicStructureMoleculeDriver(
            molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
        )
        es_problem = ElectronicStructureProblem(driver, transformers=[freeze_core_transformer])
        qubit_conv = QubitConverter(
            mapper=ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto"
        )
        qubit_conv.convert(
            es_problem.second_q_ops()[es_problem.main_property_name],
            num_particles=es_problem.num_particles,
            sector_locator=es_problem.symmetry_sector_locator,
        )
        self.assertListEqual(qubit_conv.z2symmetries.tapering_values, [-1, 1])
