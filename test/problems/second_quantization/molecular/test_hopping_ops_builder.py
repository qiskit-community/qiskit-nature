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
from test.problems.second_quantization.molecular.resources.resource_reader import read_expected_file
import numpy as np
from qiskit_nature.operators import FermionicOp
from qiskit_nature.problems.second_quantization.molecular import fermionic_op_builder
from qiskit_nature.drivers import HDF5Driver


class TestHoppingOpsBuilder(QiskitNatureTestCase):
    """Tests Hopping Operators builder."""

    def test_build_hopping_operators(self):
        """Tests that the correct hopping operator is built from QMolecule."""
        pass  # TODO
