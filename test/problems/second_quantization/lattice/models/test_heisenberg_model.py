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

"""Test HeisenbergModel."""

from typing import cast
from test import QiskitNatureTestCase

import numpy as np
from numpy.testing import assert_array_equal
from retworkx import PyGraph, is_isomorphic

from qiskit_nature.problems.second_quantization.lattice import HeisenbergModel, IsingModel, Lattice


class TestHeisenbergModel(QiskitNatureTestCase):
    """TestHeisenbergModel"""

    def test_init(self):
        """Test init."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(2))
        weighted_edge_list = [(0, 1, 1.0)]
        graph.add_edges_from(weighted_edge_list)
        ism_graph = PyGraph(multigraph=False)
        ism_graph.add_nodes_from(range(2))
        ism_weighted_edge_list = [(0, 1, 1.0), (0, 0, 1.0), (1, 1, 1.0)]
        ism_graph.add_edges_from(ism_weighted_edge_list)
        lattice = Lattice(graph)
        ism_lattice = Lattice(ism_graph)
        hm = HeisenbergModel(lattice)
        hm_to_ism = HeisenbergModel(ism_lattice)
        ism = IsingModel(ism_lattice)

        with self.subTest("Check the graph."):
            self.assertTrue(
                is_isomorphic(hm.lattice.graph, lattice.graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the second q op representation."):
            coupling = [("X_0 X_1", -1.0), ("Y_0 Y_1", -1.0), ("Z_0 Z_1", -1.0)]

            hamiltonian = coupling

            self.assertSetEqual(set(hamiltonian), set(hm.second_q_ops().to_list()))

        with self.subTest(
            "Check if, in a special case, the second q op produced by HeisenbergModel matches with those produced by IsingModel"
        ):
            model_constants = {"J_x": 0, "J_y": 0, "J_z": 1, "h": 1}
            ext_magnetic_field = {"B_x": True, "B_y": False, "B_z": False}

            self.assertSetEqual(
                set(ism.second_q_ops().to_list()),
                set(hm_to_ism.second_q_ops(model_constants, ext_magnetic_field).to_list()),
            )
