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
from test import QiskitNatureTestCase, requires_extra_library
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals

from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.problems.second_quantization.electronic.builders.hopping_ops_builder import (
    _build_qeom_hopping_ops,
)


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    @requires_extra_library
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

        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = ElectronicStructureProblem(self.driver)
        self.electronic_structure_problem.second_q_ops()
        self.particle_number = (
            self.electronic_structure_problem.grouped_property_transformed.get_property(
                "ParticleNumber"
            )
        )

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built from QMolecule."""
        # TODO extract it somewhere
        expected_hopping_operators = (
            {
<<<<<<< HEAD
                "E_0": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, False, False, False, False, False, False],
                            [True, True, False, False, False, True, False, False],
                            [True, True, False, False, True, False, False, False],
                            [True, True, False, False, True, True, False, False],
                        ],
                        coeffs=[1.0 + 0.0j, 0.0 - 1.0j, 0.0 + 1.0j, 1.0 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "Edag_0": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, False, False, False, False, False, False],
                            [True, True, False, False, False, True, False, False],
                            [True, True, False, False, True, False, False, False],
                            [True, True, False, False, True, True, False, False],
                        ],
                        coeffs=[-1.0 + 0.0j, 0.0 - 1.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "E_1": PauliSumOp(
                    SparsePauliOp(
                        [
                            [False, False, True, True, False, False, False, False],
                            [False, False, True, True, False, False, False, True],
                            [False, False, True, True, False, False, True, False],
                            [False, False, True, True, False, False, True, True],
                        ],
                        coeffs=[1.0 + 0.0j, 0.0 - 1.0j, 0.0 + 1.0j, 1.0 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "Edag_1": PauliSumOp(
                    SparsePauliOp(
                        [
                            [False, False, True, True, False, False, False, False],
                            [False, False, True, True, False, False, False, True],
                            [False, False, True, True, False, False, True, False],
                            [False, False, True, True, False, False, True, True],
                        ],
                        coeffs=[-1.0 + 0.0j, 0.0 - 1.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                    ),
                    coeff=1.0,
                ),
                "E_2": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, True, True, False, False, False, False],
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
                            [True, True, True, True, True, True, True, True],
                        ],
                        coeffs=[
                            1.0 + 0.0j,
                            0.0 - 1.0j,
                            0.0 + 1.0j,
                            1.0 + 0.0j,
                            0.0 - 1.0j,
                            -1.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 - 1.0j,
                            0.0 + 1.0j,
                            1.0 + 0.0j,
                            -1.0 + 0.0j,
                            0.0 + 1.0j,
                            1.0 + 0.0j,
                            0.0 - 1.0j,
                            0.0 + 1.0j,
                            1.0 + 0.0j,
                        ],
                    ),
                    coeff=1.0,
                ),
                "Edag_2": PauliSumOp(
                    SparsePauliOp(
                        [
                            [True, True, True, True, False, False, False, False],
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
                            [True, True, True, True, True, True, True, True],
                        ],
                        coeffs=[
                            1.0 + 0.0j,
                            0.0 + 1.0j,
                            0.0 - 1.0j,
                            1.0 + 0.0j,
                            0.0 + 1.0j,
                            -1.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 1.0j,
                            0.0 - 1.0j,
                            1.0 + 0.0j,
                            -1.0 + 0.0j,
                            0.0 - 1.0j,
                            1.0 + 0.0j,
                            0.0 + 1.0j,
                            0.0 - 1.0j,
                            1.0 + 0.0j,
                        ],
                    ),
                    coeff=1.0,
=======
                "E_0": PauliSumOp.from_list(
                    [("IIXX", 1), ("IIYX", 1j), ("IIXY", -1j), ("IIYY", 1)]
                ),
                "Edag_0": PauliSumOp.from_list(
                    [("IIXX", -1), ("IIYX", 1j), ("IIXY", -1j), ("IIYY", -1)]
                ),
                "E_1": PauliSumOp.from_list(
                    [("XXII", 1), ("YXII", 1j), ("XYII", -1j), ("YYII", 1)]
                ),
                "Edag_1": PauliSumOp.from_list(
                    [("XXII", -1), ("YXII", 1j), ("XYII", -1j), ("YYII", -1)]
                ),
                "E_2": PauliSumOp.from_list(
                    [
                        ("XXXX", 1),
                        ("YXXX", 1j),
                        ("XYXX", -1j),
                        ("YYXX", 1),
                        ("XXYX", 1j),
                        ("YXYX", -1),
                        ("XYYX", 1),
                        ("YYYX", 1j),
                        ("XXXY", -1j),
                        ("YXXY", 1),
                        ("XYXY", -1),
                        ("YYXY", -1j),
                        ("XXYY", 1),
                        ("YXYY", 1j),
                        ("XYYY", -1j),
                        ("YYYY", 1),
                    ]
                ),
                "Edag_2": PauliSumOp.from_list(
                    [
                        ("XXXX", 1),
                        ("YXXX", -1j),
                        ("XYXX", 1j),
                        ("YYXX", 1),
                        ("XXYX", -1j),
                        ("YXYX", -1),
                        ("XYYX", 1),
                        ("YYYX", -1j),
                        ("XXXY", 1j),
                        ("YXXY", 1),
                        ("XYXY", -1),
                        ("YYXY", 1j),
                        ("XXYY", 1),
                        ("YXYY", -1j),
                        ("XYYY", 1j),
                        ("YYYY", 1),
                    ]
>>>>>>> 3e66f54 (Fix mapper (#357))
                ),
            },
            {"E_0": [], "Edag_0": [], "E_1": [], "Edag_1": [], "E_2": [], "Edag_2": []},
            {
                "E_0": ((0,), (1,)),
                "Edag_0": ((1,), (0,)),
                "E_1": ((2,), (3,)),
                "Edag_1": ((3,), (2,)),
                "E_2": ((0, 2), (1, 3)),
                "Edag_2": ((1, 3), (0, 2)),
            },
        )

        hopping_operators = _build_qeom_hopping_ops(self.particle_number, self.qubit_converter)
        self.assertEqual(hopping_operators, expected_hopping_operators)
