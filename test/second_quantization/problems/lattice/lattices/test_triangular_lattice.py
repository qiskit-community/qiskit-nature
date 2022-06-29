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

"""Test TriangularLattice."""
from test import QiskitNatureTestCase
from numpy.testing import assert_array_equal
import numpy as np
from retworkx import PyGraph, is_isomorphic
from qiskit_nature.second_q.problems.lattice import (
    BoundaryCondition,
    TriangularLattice,
)


class TestTriangularLattice(QiskitNatureTestCase):
    """Test TriangularLattice."""

    def test_init_open(self):
        """Test init for the open boundary conditions."""
        rows = 3
        cols = 2
        edge_parameter = (1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j)
        onsite_parameter = 0.0
        boundary_condition = BoundaryCondition.OPEN
        triangular = TriangularLattice(
            rows, cols, edge_parameter, onsite_parameter, boundary_condition
        )
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(6))
            weighted_list = [
                (0, 1, 1.0 + 1.0j),
                (1, 2, 1.0 + 1.0j),
                (3, 4, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (0, 3, 2.0 + 2.0j),
                (1, 4, 2.0 + 2.0j),
                (2, 5, 2.0 + 2.0j),
                (0, 4, 3.0 + 3.0j),
                (1, 5, 3.0 + 3.0j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
            ]
            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(triangular.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(triangular.num_nodes, 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(triangular.node_indexes), set(range(6)))

        with self.subTest("Check the set of weights."):
            target_set = {
                (0, 1, 1.0 + 1.0j),
                (1, 2, 1.0 + 1.0j),
                (3, 4, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (0, 3, 2.0 + 2.0j),
                (1, 4, 2.0 + 2.0j),
                (2, 5, 2.0 + 2.0j),
                (0, 4, 3.0 + 3.0j),
                (1, 5, 3.0 + 3.0j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
            }
            self.assertSetEqual(set(triangular.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.array(
                [
                    [0.0, 1.0 + 1.0j, 0.0, 2.0 + 2.0j, 3.0 + 3.0j, 0],
                    [1.0 - 1.0j, 0.0, 1.0 + 1.0j, 0.0, 2.0 + 2.0j, 3.0 + 3.0j],
                    [0.0, 1.0 - 1.0j, 0.0, 0.0, 0.0, 2.0 + 2.0j],
                    [2.0 - 2.0j, 0.0, 0.0, 0.0, 1.0 + 1.0j, 0.0],
                    [3.0 - 3.0j, 2.0 - 2.0j, 0.0, 1.0 - 1.0j, 0.0, 1.0 + 1.0j],
                    [0.0, 3.0 - 3.0j, 2.0 - 2.0j, 0.0, 1.0 - 1.0j, 0.0],
                ]
            )
            assert_array_equal(triangular.to_adjacency_matrix(weighted=True), target_matrix)

    def test_init_periodic(self):
        """Test init for the periodic boundary conditions."""
        rows = 3
        cols = 2
        edge_parameter = (1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j)
        onsite_parameter = 0.0
        boundary_condition = BoundaryCondition.PERIODIC
        triangular = TriangularLattice(
            rows, cols, edge_parameter, onsite_parameter, boundary_condition
        )
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(6))
            weighted_list = [
                (0, 1, 1.0 + 1.0j),
                (1, 2, 1.0 + 1.0j),
                (0, 2, 1.0 - 1.0j),
                (3, 4, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (3, 5, 1.0 - 1.0j),
                (0, 3, 2.0 + 2.0j),
                (1, 4, 2.0 + 2.0j),
                (2, 5, 2.0 + 2.0j),
                (0, 4, 3.0 + 3.0j),
                (1, 5, 3.0 + 3.0j),
                (2, 3, 3.0 + 3.0j),
                (1, 3, 3.0 - 3.0j),
                (2, 4, 3.0 - 3.0j),
                (0, 5, 3.0 - 3.0j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
            ]
            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(triangular.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(triangular.num_nodes, 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(triangular.node_indexes), set(range(6)))

        with self.subTest("Check the set of weights."):
            target_set = {
                (0, 1, 1.0 + 1.0j),
                (1, 2, 1.0 + 1.0j),
                (0, 2, 1.0 - 1.0j),
                (3, 4, 1.0 + 1.0j),
                (4, 5, 1.0 + 1.0j),
                (3, 5, 1.0 - 1.0j),
                (0, 3, 2.0 + 2.0j),
                (1, 4, 2.0 + 2.0j),
                (2, 5, 2.0 + 2.0j),
                (0, 4, 3.0 + 3.0j),
                (1, 5, 3.0 + 3.0j),
                (2, 3, 3.0 + 3.0j),
                (1, 3, 3.0 - 3.0j),
                (2, 4, 3.0 - 3.0j),
                (0, 5, 3.0 - 3.0j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
            }
            self.assertSetEqual(set(triangular.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.array(
                [
                    [0.0, 1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 3.0 - 3.0j],
                    [1.0 - 1.0j, 0.0, 1.0 + 1.0j, 3.0 - 3.0j, 2.0 + 2.0j, 3.0 + 3.0j],
                    [1.0 + 1.0j, 1.0 - 1.0j, 0.0, 3.0 + 3.0j, 3.0 - 3.0j, 2.0 + 2.0j],
                    [2.0 - 2.0j, 3.0 + 3.0j, 3.0 - 3.0j, 0.0, 1.0 + 1.0j, 1.0 - 1.0j],
                    [3.0 - 3.0j, 2.0 - 2.0j, 3.0 + 3.0j, 1.0 - 1.0j, 0.0, 1.0 + 1.0j],
                    [3.0 + 3.0j, 3.0 - 3.0j, 2.0 - 2.0j, 1.0 + 1.0j, 1.0 - 1.0j, 0.0],
                ]
            )
            assert_array_equal(triangular.to_adjacency_matrix(weighted=True), target_matrix)
