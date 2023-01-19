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

"""Test KagomeLattice."""
from test import QiskitNatureTestCase
from numpy.testing import assert_array_equal
import numpy as np
from rustworkx import PyGraph, is_isomorphic
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    KagomeLattice,
)


class TestKagomeLattice(QiskitNatureTestCase):
    """Test KagomeLattice."""

    def test_init_open(self):
        """Test init for the open boundary conditions."""
        rows = 3
        cols = 2
        num_sites_per_cell = 3
        edge_parameter = 1 - 1j
        onsite_parameter = 0.0
        boundary_condition = BoundaryCondition.OPEN
        kagome = KagomeLattice(rows, cols, edge_parameter, onsite_parameter, boundary_condition)
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(num_sites_per_cell * 6))
            weighted_list = [
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
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
                (6, 6, 0.0),
                (7, 7, 0.0),
                (8, 8, 0.0),
                (9, 9, 0.0),
                (10, 10, 0.0),
                (11, 11, 0.0),
                (12, 12, 0.0),
                (13, 13, 0.0),
                (14, 14, 0.0),
                (15, 15, 0.0),
                (16, 16, 0.0),
                (17, 17, 0.0),
            ]
            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(kagome.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(kagome.num_nodes, num_sites_per_cell * 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(kagome.node_indexes), set(range(num_sites_per_cell * 6)))

        with self.subTest("Check the set of weights."):
            target_set = {
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
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
                (6, 6, 0.0),
                (7, 7, 0.0),
                (8, 8, 0.0),
                (9, 9, 0.0),
                (10, 10, 0.0),
                (11, 11, 0.0),
                (12, 12, 0.0),
                (13, 13, 0.0),
                (14, 14, 0.0),
                (15, 15, 0.0),
                (16, 16, 0.0),
                (17, 17, 0.0),
            }
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
        rows = 3
        cols = 2
        edge_parameter = 1 - 1j
        num_sites_per_cell = 3
        onsite_parameter = 0.0
        boundary_condition = BoundaryCondition.PERIODIC
        kagome = KagomeLattice(rows, cols, edge_parameter, onsite_parameter, boundary_condition)
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(num_sites_per_cell * 6))
            weighted_list = [
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
                (7, 0, 1 + 1j),
                (11, 7, 1 + 1j),
                (16, 9, 1 + 1j),
                (11, 0, 1 + 1j),
                (14, 1, 1 + 1j),
                (14, 3, 1 + 1j),
                (17, 4, 1 + 1j),
                (17, 6, 1 + 1j),
                (2, 16, 1 + 1j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
                (6, 6, 0.0),
                (7, 7, 0.0),
                (8, 8, 0.0),
                (9, 9, 0.0),
                (10, 10, 0.0),
                (11, 11, 0.0),
                (12, 12, 0.0),
                (13, 13, 0.0),
                (14, 14, 0.0),
                (15, 15, 0.0),
                (16, 16, 0.0),
                (17, 17, 0.0),
            ]
            target_graph.add_edges_from(weighted_list)
            self.assertTrue(
                is_isomorphic(kagome.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(kagome.num_nodes, num_sites_per_cell * 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(kagome.node_indexes), set(range(num_sites_per_cell * 6)))

        with self.subTest("Check the set of weights."):
            target_set = {
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
                (7, 0, 1 + 1j),
                (11, 7, 1 + 1j),
                (16, 9, 1 + 1j),
                (11, 0, 1 + 1j),
                (14, 1, 1 + 1j),
                (14, 3, 1 + 1j),
                (17, 4, 1 + 1j),
                (17, 6, 1 + 1j),
                (2, 16, 1 + 1j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
                (3, 3, 0.0),
                (4, 4, 0.0),
                (5, 5, 0.0),
                (6, 6, 0.0),
                (7, 7, 0.0),
                (8, 8, 0.0),
                (9, 9, 0.0),
                (10, 10, 0.0),
                (11, 11, 0.0),
                (12, 12, 0.0),
                (13, 13, 0.0),
                (14, 14, 0.0),
                (15, 15, 0.0),
                (16, 16, 0.0),
                (17, 17, 0.0),
            }
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