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

""" Test Numerical qEOM excited states calculation."""

import unittest

from test import QiskitNatureTestCase
from ddt import ddt, named_data
from typing import List, Dict

import numpy as np

from qiskit.algorithms.eigensolvers import NumPyEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    VQEUCCFactory,
    QEOM,
)
import qiskit_nature.optionals as _optionals
from .resources.expected_transition_amplitudes import reference_trans_amps

@ddt
class TestNumericalQEOMObscalculation(QiskitNatureTestCase):
    """Test qEOM excited state observables calculation."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 8
        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.75",
            unit=DistanceUnit.ANGSTROM,
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

        self.reference_trans_amps = reference_trans_amps
        self.electronic_structure_problem = self.driver.run()
        hamiltonian_op, _ = self.electronic_structure_problem.second_q_ops()
        self.aux_ops = {"hamiltonian": hamiltonian_op}

    def _assert_energies(self, computed, references, *, places=4):
        with self.subTest("same number of energies"):
            self.assertEqual(len(computed), len(references))

        with self.subTest("ground state"):
            self.assertAlmostEqual(computed[0], references[0], places=places)

        for i in range(1, len(computed)):
            with self.subTest(f"{i}. excited state"):
                self.assertAlmostEqual(computed[i], references[i], places=places)

    def _assert_transition_amplitudes(self, computed, references, *, places=4):
        with self.subTest("same number of indices"):
            self.assertEqual(len(computed), len(references))

        with self.subTest("operators computed are reference operators"):
            self.assertEqual(computed.keys(), references.keys())

        with self.subTest("same transition amplitude absolute value"):
            for key in computed.keys():
                for opkey in computed[key].keys():
                    trans_amp = np.abs(computed[key][opkey][0])
                    trans_amp_expected = np.abs(references[key][opkey][0])
                    self.assertAlmostEqual(trans_amp, trans_amp_expected, places=places)

    @named_data(
        ["JWM", QubitConverter(JordanWignerMapper())],
        ["JWM_Z2", QubitConverter(JordanWignerMapper(), z2symmetry_reduction="auto")],
        ["PM", QubitConverter(ParityMapper())],
        ["PM_TQR", QubitConverter(ParityMapper(), two_qubit_reduction=True)],
        ["PM_Z2", QubitConverter(ParityMapper(), z2symmetry_reduction="auto")],
        ["PM_TQR_Z2", QubitConverter(ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto")],
        ["BKM", QubitConverter(BravyiKitaevMapper())],
        ["BKM_Z2", QubitConverter(BravyiKitaevMapper(), z2symmetry_reduction="auto")],
    )
    def test_aux_ops_qeom(self, converter: QubitConverter):
        estimator = Estimator()
        solver = VQEUCCFactory(estimator, UCCSD(), SLSQP())
        gsc = GroundStateEigensolver(converter, solver)
        esc = QEOM(gsc, estimator, "sd", aux_eval_rules="diag")
        results = esc.solve(self.electronic_structure_problem, aux_operators=self.aux_ops)

        energies_recalculated = np.zeros_like(results.computed_energies)
        for estate, aux_op in enumerate(results.aux_operators_evaluated):
            energies_recalculated[estate] = aux_op["hamiltonian"]

        self._assert_energies(results.computed_energies, self.reference_energies)
        self._assert_energies(energies_recalculated, self.reference_energies)

    @named_data(
        ["JWM", QubitConverter(JordanWignerMapper())],
        ["JWM_Z2", QubitConverter(JordanWignerMapper(), z2symmetry_reduction="auto")],
        ["PM", QubitConverter(ParityMapper())],
        ["PM_TQR", QubitConverter(ParityMapper(), two_qubit_reduction=True)],
        ["PM_Z2", QubitConverter(ParityMapper(), z2symmetry_reduction="auto")],
        ["PM_TQR_Z2", QubitConverter(ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto")],
        ["BKM", QubitConverter(BravyiKitaevMapper())],
        ["BKM_Z2", QubitConverter(BravyiKitaevMapper(), z2symmetry_reduction="auto")],
    )
    def test_trans_amps_qeom(self, converter: QubitConverter):
        estimator = Estimator()
        solver = VQEUCCFactory(estimator, UCCSD(), SLSQP())
        gsc = GroundStateEigensolver(converter, solver)
        esc = QEOM(gsc, estimator, "sd", aux_eval_rules="all")
        results = esc.solve(self.electronic_structure_problem, aux_operators=self.aux_ops)

        transition_amplitudes = results.raw_result.transition_amplitudes
        print(transition_amplitudes)

        self._assert_transition_amplitudes(
            transition_amplitudes, self.reference_trans_amps, places=4
        )   


if __name__ == "__main__":
    unittest.main()
