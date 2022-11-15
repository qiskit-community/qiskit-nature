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

""" Test Methods FCIDump """

import unittest

from typing import List, Optional

from test.second_q.drivers.test_driver_methods_gsc import TestDriverMethods
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.transformers import BaseTransformer, ActiveSpaceTransformer
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.mappers import QubitConverter, JordanWignerMapper
from qiskit_nature.second_q.problems import EigenstateResult


class TestMethodsFCIDump(TestDriverMethods):
    """Methods FCIDump tests"""

    @staticmethod
    def _run_fcidump(
        fcidump: FCIDump,
        converter: QubitConverter = QubitConverter(JordanWignerMapper()),
        transformers: Optional[List[BaseTransformer]] = None,
    ) -> EigenstateResult:
        problem: BaseProblem = fcidump_to_problem(fcidump)

        if transformers is not None:
            for trafo in transformers:
                problem = trafo.transform(problem)

        solver = NumPyMinimumEigensolver()

        gsc = GroundStateEigensolver(converter, solver)

        result = gsc.solve(problem)
        return result

    def test_lih(self):
        """LiH test"""
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_lih.fcidump", "second_q/formats/fcidump")
        )
        result = self._run_fcidump(fcidump)
        self._assert_energy(result, "lih")

    def test_oh(self):
        """OH test"""
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_oh.fcidump", "second_q/formats/fcidump")
        )
        result = self._run_fcidump(fcidump)
        self._assert_energy(result, "oh")

    def test_lih_with_active_space(self):
        """LiH with active space test"""
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_lih.fcidump", "second_q/formats/fcidump"),
        )
        result = self._run_fcidump(fcidump, transformers=[ActiveSpaceTransformer(4, 6)])
        self._assert_energy(result, "lih")

    def test_oh_with_active_space(self):
        """OH with active space test"""
        fcidump = FCIDump.from_file(
            self.get_resource_path("test_fcidump_oh.fcidump", "second_q/formats/fcidump"),
        )
        result = self._run_fcidump(fcidump, transformers=[ActiveSpaceTransformer((5, 4), 6)])
        self._assert_energy(result, "oh")


if __name__ == "__main__":
    unittest.main()
