# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test KagomeLattice."""
import unittest
from test import QiskitNatureTestCase
from numpy.testing import assert_array_equal
import numpy as np
from rustworkx import PyGraph, is_isomorphic  # type: ignore[attr-defined]
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    KagomeLattice,
)


class TestKagomeLattice(QiskitNatureTestCase):
    """Test KagomeLattice."""

    def setUp(self):
        super().setUp()

        self.rows = 3
        self.cols = 2
        self.num_sites_per_cell = 3
        self.edge_parameter = 1 - 1j
        self.onsite_parameter = 0.0

        self.weighted_bulk_edges = [
            (2, 0, 1 - 1j),
            (0, 1, 1 - 1j),
            (1, 2, 1 - 1j),
            (3, 4, 1 - 1j),
            (5, 3, 1 - 1j),
            (4, 5, 1 - 1j),
            (6, 7, 1 - 1j),
            (8, 6, 1 - 1j),
            (7, 8, 1 - 1j),
            (9, 10, 1 - 1j),
            (11, 9, 1 - 1j),
            (10, 11, 1 - 1j),
            (12, 13, 1 - 1j),
            (14, 12, 1 - 1j),
            (13, 14, 1 - 1j),
            (15, 16, 1 - 1j),
            (17, 15, 1 - 1j),
            (16, 17, 1 - 1j),
            (1, 3, 1 - 1j),
            (2, 9, 1 - 1j),
            (5, 10, 1 - 1j),
            (5, 12, 1 - 1j),
            (4, 6, 1 - 1j),
            (8, 13, 1 - 1j),
            (8, 15, 1 - 1j),
            (10, 12, 1 - 1j),
            (13, 15, 1 - 1j),
        ]

        self.weighted_self_loops = [(i, i, 0.0) for i in range(18)]

        self.weighted_boundary_x_edges = [
            (7, 0, 1 + 1j),
            (16, 9, 1 + 1j),
        ]

        self.weighted_boundary_y_edges = [
            (11, 0, 1 + 1j),
            (14, 3, 1 + 1j),
            (17, 6, 1 + 1j),
        ]

        self.weighted_boundary_xy_edges = [
            (14, 1, 1 + 1j),
            (17, 4, 1 + 1j),
            (2, 16, 1 + 1j),
            (11, 7, 1 + 1j),
        ]

    def test_init_open(self):
        """Test init for the open boundary conditions."""
        boundary_condition = BoundaryCondition.OPEN
        kagome = KagomeLattice(
            self.rows, self.cols, self.edge_parameter, self.onsite_parameter, boundary_condition
        )
        weighted_list = self.weighted_bulk_edges + self.weighted_self_loops

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(self.num_sites_per_cell * 6))
            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(kagome.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(kagome.num_nodes, self.num_sites_per_cell * 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(kagome.node_indexes), set(range(self.num_sites_per_cell * 6)))

        with self.subTest("Check the set of weights."):
            target_set = set(weighted_list)
            self.assertSetEqual(set(kagome.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.zeros((kagome.num_nodes, kagome.num_nodes), dtype=np.complex128)

            # Fill in the edges from the edge list
            for edge in weighted_list:
                i, j, weight = edge
                if j > i:
                    target_matrix[i][j] = weight
                    target_matrix[j][i] = weight.conjugate()
                elif i > j:
                    target_matrix[j][i] = weight
                    target_matrix[i][j] = weight.conjugate()
                else:
                    target_matrix[i][i] = weight

            assert_array_equal(kagome.to_adjacency_matrix(weighted=True), target_matrix)

    def test_init_periodic(self):
        """Test init for the periodic boundary conditions."""
        boundary_condition = BoundaryCondition.PERIODIC
        kagome = KagomeLattice(
            self.rows, self.cols, self.edge_parameter, self.onsite_parameter, boundary_condition
        )
        weighted_list = (
            self.weighted_bulk_edges
            + self.weighted_self_loops
            + self.weighted_boundary_x_edges
            + self.weighted_boundary_y_edges
            + self.weighted_boundary_xy_edges
        )

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(self.num_sites_per_cell * 6))

            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(kagome.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(kagome.num_nodes, self.num_sites_per_cell * 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(kagome.node_indexes), set(range(self.num_sites_per_cell * 6)))

        with self.subTest("Check the set of weights."):
            target_set = set(weighted_list)
            self.assertSetEqual(set(kagome.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.zeros((kagome.num_nodes, kagome.num_nodes), dtype=np.complex128)

            # Fill in the edges from the edge list
            for edge in weighted_list:
                i, j, weight = edge
                if j > i:
                    target_matrix[i][j] = weight
                    target_matrix[j][i] = weight.conjugate()
                elif i > j:
                    target_matrix[j][i] = weight
                    target_matrix[i][j] = weight.conjugate()
                else:
                    target_matrix[i][i] = weight

            assert_array_equal(kagome.to_adjacency_matrix(weighted=True), target_matrix)

    def test_init_x_periodic(self):
        """Test init for the periodic boundary conditions."""
        boundary_condition = (BoundaryCondition.PERIODIC, BoundaryCondition.OPEN)
        kagome = KagomeLattice(
            self.rows, self.cols, self.edge_parameter, self.onsite_parameter, boundary_condition
        )
        weighted_list = (
            self.weighted_bulk_edges + self.weighted_self_loops + self.weighted_boundary_x_edges
        )

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(self.num_sites_per_cell * 6))

            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(kagome.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(kagome.num_nodes, self.num_sites_per_cell * 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(kagome.node_indexes), set(range(self.num_sites_per_cell * 6)))

        with self.subTest("Check the set of weights."):
            target_set = set(weighted_list)
            self.assertSetEqual(set(kagome.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.zeros((kagome.num_nodes, kagome.num_nodes), dtype=np.complex128)

            # Fill in the edges from the edge list
            for edge in weighted_list:
                i, j, weight = edge
                if j > i:
                    target_matrix[i][j] = weight
                    target_matrix[j][i] = weight.conjugate()
                elif i > j:
                    target_matrix[j][i] = weight
                    target_matrix[i][j] = weight.conjugate()
                else:
                    target_matrix[i][i] = weight

            assert_array_equal(kagome.to_adjacency_matrix(weighted=True), target_matrix)

    def test_init_y_periodic(self):
        """Test init for the periodic boundary conditions."""
        boundary_condition = (BoundaryCondition.OPEN, BoundaryCondition.PERIODIC)
        kagome = KagomeLattice(
            self.rows, self.cols, self.edge_parameter, self.onsite_parameter, boundary_condition
        )
        weighted_list = (
            self.weighted_bulk_edges + self.weighted_self_loops + self.weighted_boundary_y_edges
        )

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(self.num_sites_per_cell * 6))

            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(kagome.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(kagome.num_nodes, self.num_sites_per_cell * 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(kagome.node_indexes), set(range(self.num_sites_per_cell * 6)))

        with self.subTest("Check the set of weights."):
            target_set = set(weighted_list)
            self.assertSetEqual(set(kagome.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.zeros((kagome.num_nodes, kagome.num_nodes), dtype=np.complex128)

            # Fill in the edges from the edge list
            for edge in weighted_list:
                i, j, weight = edge
                if j > i:
                    target_matrix[i][j] = weight
                    target_matrix[j][i] = weight.conjugate()
                elif i > j:
                    target_matrix[j][i] = weight
                    target_matrix[i][j] = weight.conjugate()
                else:
                    target_matrix[i][i] = weight

            assert_array_equal(kagome.to_adjacency_matrix(weighted=True), target_matrix)


if __name__ == "__main__":
    unittest.main()
