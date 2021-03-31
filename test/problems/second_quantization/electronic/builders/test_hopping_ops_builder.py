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

"""Tests Hopping Operators builder."""
from test import QiskitNatureTestCase
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.problems.second_quantization.electronic.builders.hopping_ops_builder import \
    _build_qeom_hopping_ops


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.75',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitNatureError:
            self.skipTest('PYSCF driver does not appear to be installed')

        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = ElectronicStructureProblem(self.driver)
        self.electronic_structure_problem.second_q_ops()
        self.q_molecule = self.electronic_structure_problem.molecule_data

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built from QMolecule."""
        # TODO extract it somewhere
        expected_hopping_operators = (
            {'E_0': PauliSumOp(
                SparsePauliOp([[True, True, False, False, False, False, False, False],
                               [True, True, False, False, False, True, False, False],
                               [True, True, False, False, True, False, False, False],
                               [True, True, False, False, True, True, False, False]],
                              coeffs=[1. + 0.j, 0. - 1.j, 0. + 1.j, 1. + 0.j]),
                coeff=1.0),
                'Edag_0': PauliSumOp(
                    SparsePauliOp([[True, True, False, False, False, False, False, False],
                                   [True, True, False, False, False, True, False, False],
                                   [True, True, False, False, True, False, False, False],
                                   [True, True, False, False, True, True, False, False]],
                                  coeffs=[-1. + 0.j, 0. - 1.j, 0. + 1.j, -1. + 0.j]),
                    coeff=1.0),
                'E_1': PauliSumOp(
                    SparsePauliOp([[False, False, True, True, False, False, False, False],
                                   [False, False, True, True, False, False, False, True],
                                   [False, False, True, True, False, False, True, False],
                                   [False, False, True, True, False, False, True, True]],
                                  coeffs=[1. + 0.j, 0. - 1.j, 0. + 1.j, 1. + 0.j]),
                    coeff=1.0),
                'Edag_1': PauliSumOp(
                    SparsePauliOp([[False, False, True, True, False, False, False, False],
                                   [False, False, True, True, False, False, False, True],
                                   [False, False, True, True, False, False, True, False],
                                   [False, False, True, True, False, False, True, True]],
                                  coeffs=[-1. + 0.j, 0. - 1.j, 0. + 1.j, -1. + 0.j]),
                    coeff=1.0),
                'E_2': PauliSumOp(
                    SparsePauliOp([[True, True, True, True, False, False, False, False],
                                   [True, True, True, True, False, False, False, True],
                                   [True, True, True, True, False, False, True, False],
                                   [True, True, True, True, False, False, True, True],
                                   [True, True, True, True, False, True, False, False],
                                   [True, True, True, True, False, True, False, True],
                                   [True, True, True, True, False, True, True, False],
                                   [True, True, True, True, False, True, True, True],
                                   [True, True, True, True, True, False, False, False],
                                   [True, True, True, True, True, False, False, True],
                                   [True, True, True, True, True, False, True, False],
                                   [True, True, True, True, True, False, True, True],
                                   [True, True, True, True, True, True, False, False],
                                   [True, True, True, True, True, True, False, True],
                                   [True, True, True, True, True, True, True, False],
                                   [True, True, True, True, True, True, True, True]],
                                  coeffs=[1. + 0.j, 0. - 1.j, 0. + 1.j, 1. + 0.j,
                                          0. - 1.j,
                                          -1. + 0.j, 1. + 0.j, 0. - 1.j,
                                          0. + 1.j, 1. + 0.j, -1. + 0.j, 0. + 1.j,
                                          1. + 0.j,
                                          0. - 1.j, 0. + 1.j, 1. + 0.j]), coeff=1.0),
                'Edag_2': PauliSumOp(
                    SparsePauliOp([[True, True, True, True, False, False, False, False],
                                   [True, True, True, True, False, False, False, True],
                                   [True, True, True, True, False, False, True, False],
                                   [True, True, True, True, False, False, True, True],
                                   [True, True, True, True, False, True, False, False],
                                   [True, True, True, True, False, True, False, True],
                                   [True, True, True, True, False, True, True, False],
                                   [True, True, True, True, False, True, True, True],
                                   [True, True, True, True, True, False, False, False],
                                   [True, True, True, True, True, False, False, True],
                                   [True, True, True, True, True, False, True, False],
                                   [True, True, True, True, True, False, True, True],
                                   [True, True, True, True, True, True, False, False],
                                   [True, True, True, True, True, True, False, True],
                                   [True, True, True, True, True, True, True, False],
                                   [True, True, True, True, True, True, True, True]],
                                  coeffs=[1. + 0.j, 0. + 1.j, 0. - 1.j, 1. + 0.j,
                                          0. + 1.j, -1. + 0.j, 1. + 0.j, 0. + 1.j,
                                          0. - 1.j, 1. + 0.j, -1. + 0.j, 0. - 1.j,
                                          1. + 0.j, 0. + 1.j, 0. - 1.j, 1. + 0.j]),
                    coeff=1.0)},
            {'E_0': [], 'Edag_0': [], 'E_1': [], 'Edag_1': [], 'E_2': [], 'Edag_2': []},
            {'E_0': ((0,), (1,)), 'Edag_0': ((1,), (0,)), 'E_1': ((2,), (3,)),
             'Edag_1': ((3,), (2,)), 'E_2': ((0, 2), (1, 3)), 'Edag_2': ((1, 3), (0, 2))})

        hopping_operators = _build_qeom_hopping_ops(self.q_molecule,
                                                    self.qubit_converter)
        self.assertEqual(hopping_operators, expected_hopping_operators)
