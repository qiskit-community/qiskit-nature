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

"""Test LineLattice."""
from test import QiskitNatureTestCase
from numpy.testing import assert_array_equal
import numpy as np
from retworkx import PyGraph, is_isomorphic
from qiskit_nature.problems.second_quantization.lattice.lattice.line_lattice import LineLattice


class TestLineLattice(QiskitNatureTestCase):
    """Test LineLattice."""

    def test_init(self):
        """Test init."""
        num_nodes = 6
        edge_parameter = 1.0 + 1.0j
        onsite_parameter = 1.0
        boundary_condition = "open"
        linelattice = LineLattice(num_nodes, edge_parameter, onsite_parameter, boundary_condition)
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(6))
            weighted_edge_list = [
                (0, 1, 1.0 + 1.0j),
                (1, 2, 1.0 + 1.0j),
                (2, 3, 1.0 + 1.0j),
                (3, 4, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (0, 0, 1.0),
                (1, 1, 1.0),
                (2, 2, 1.0),
                (3, 3, 1.0),
                (4, 4, 1.0),
                (5, 5, 1.0),
            ]
            target_graph.add_edges_from(weighted_edge_list)
            self.assertTrue(
                is_isomorphic(linelattice.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(linelattice.num_nodes, 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(linelattice.node_indexes), set(range(6)))

        with self.subTest("Check the set of weights."):
            target_set = {
                (0, 1, 1.0 + 1.0j),
                (1, 2, 1.0 + 1.0j),
                (2, 3, 1.0 + 1.0j),
                (3, 4, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (0, 0, 1.0),
                (1, 1, 1.0),
                (2, 2, 1.0),
                (3, 3, 1.0),
                (4, 4, 1.0),
                (5, 5, 1.0),
            }
            self.assertSetEqual(set(linelattice.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.array(
                [
                    [1.0, 1.0 + 1.0j, 0.0, 0.0, 0.0, 0.0],
                    [1.0 - 1.0j, 1.0, 1.0 + 1.0j, 0.0, 0.0, 0.0],
                    [0.0, 1.0 - 1.0j, 1.0, 1.0 + 1.0j, 0.0, 0.0],
                    [0.0, 0.0, 1.0 - 1.0j, 1.0, 1.0 + 1.0j, 0.0],
                    [0.0, 0.0, 0.0, 1.0 - 1.0j, 1.0, 1.0 + 1.0j],
                    [0.0, 0.0, 0.0, 0.0, 1.0 - 1.0j, 1.0],
                ]
            )

            assert_array_equal(linelattice.to_adjacency_matrix(weighted=True), target_matrix)
