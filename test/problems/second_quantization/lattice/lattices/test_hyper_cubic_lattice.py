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

"""Test for HyperCubicLattice."""
from test import QiskitNatureTestCase
import numpy as np
from numpy.testing import assert_array_equal
from retworkx import PyGraph, is_isomorphic
from qiskit_nature.problems.second_quantization.lattice import (
    BoundaryCondition,
    HyperCubicLattice,
)


class TestHyperCubic(QiskitNatureTestCase):
    """Test HyperCubicLattice."""

    def test_init(self):
        """Test init."""
        size = (2, 2, 2)
        edge_parameter = (1.0 + 1.0j, 0.0, -2.0 - 2.0j)
        onsite_parameter = 5.0
        boundary_condition = (
            BoundaryCondition.OPEN,
            BoundaryCondition.PERIODIC,
            BoundaryCondition.OPEN,
        )
        hyper_cubic = HyperCubicLattice(size, edge_parameter, onsite_parameter, boundary_condition)

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(8))
            weighted_edge_list = [
                (0, 1, 1.0 + 1.0j),
                (2, 3, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (6, 7, 1.0 + 1.0j),
                (0, 2, 0.0),
                (1, 3, 0.0),
                (4, 6, 0.0),
                (5, 7, 0.0),
                (0, 4, -2.0 - 2.0j),
                (1, 5, -2.0 - 2.0j),
                (2, 6, -2.0 - 2.0j),
                (3, 7, -2.0 - 2.0j),
                (0, 0, 5.0),
                (1, 1, 5.0),
                (2, 2, 5.0),
                (3, 3, 5.0),
                (4, 4, 5.0),
                (5, 5, 5.0),
                (6, 6, 5.0),
                (7, 7, 5.0),
            ]
            target_graph.add_edges_from(weighted_edge_list)
            self.assertTrue(
                is_isomorphic(hyper_cubic.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(hyper_cubic.num_nodes, 8)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(hyper_cubic.node_indexes), set(range(8)))

        with self.subTest("Check the set of weights."):
            target_set = {
                (0, 1, 1.0 + 1.0j),
                (2, 3, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (6, 7, 1.0 + 1.0j),
                (0, 2, 0.0),
                (1, 3, 0.0),
                (4, 6, 0.0),
                (5, 7, 0.0),
                (0, 4, -2.0 - 2.0j),
                (1, 5, -2.0 - 2.0j),
                (2, 6, -2.0 - 2.0j),
                (3, 7, -2.0 - 2.0j),
                (0, 0, 5.0),
                (1, 1, 5.0),
                (2, 2, 5.0),
                (3, 3, 5.0),
                (4, 4, 5.0),
                (5, 5, 5.0),
                (6, 6, 5.0),
                (7, 7, 5.0),
            }
            self.assertSetEqual(set(hyper_cubic.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.array(
                [
                    [5.0, 1.0 + 1.0j, 0.0, 0.0, -2.0 - 2.0j, 0.0, 0.0, 0.0],
                    [1.0 - 1.0j, 5.0, 0.0, 0.0, 0.0, -2.0 - 2.0j, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 1.0 + 1.0j, 0.0, 0.0, -2.0 - 2.0j, 0.0],
                    [0.0, 0.0, 1.0 - 1.0j, 5.0, 0.0, 0.0, 0.0, -2.0 - 2.0j],
                    [-2.0 + 2.0j, 0.0, 0.0, 0.0, 5.0, 1.0 + 1.0j, 0.0, 0.0],
                    [0.0, -2.0 + 2.0j, 0.0, 0.0, 1.0 - 1.0j, 5.0, 0.0, 0.0],
                    [0.0, 0.0, -2.0 + 2.0j, 0.0, 0.0, 0.0, 5.0, 1.0 + 1.0j],
                    [0.0, 0.0, 0.0, -2.0 + 2.0j, 0.0, 0.0, 1.0 - 1.0j, 5.0],
                ]
            )

            assert_array_equal(hyper_cubic.to_adjacency_matrix(weighted=True), target_matrix)
