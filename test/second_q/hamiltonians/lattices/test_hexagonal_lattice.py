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

"""Test for HexgonalLattice."""
from test import QiskitNatureTestCase
import numpy as np
from numpy.testing import assert_array_equal
from rustworkx import PyGraph, is_isomorphic  # type: ignore[attr-defined]
from qiskit_nature.second_q.hamiltonians.lattices import HexagonalLattice


class TestHexagonalLattice(QiskitNatureTestCase):
    """Test HexagonalLattice"""

    def test_init(self):
        """Test init."""
        rows = 1
        cols = 2
        edge_parameter = 0 + 1.42j
        onsite_parameter = 1.0
        weighted_edge_list = [
            (0, 1, 1.42j),
            (1, 2, 1.42j),
            (3, 4, 1.42j),
            (4, 5, 1.42j),
            (5, 6, 1.42j),
            (7, 8, 1.42j),
            (8, 9, 1.42j),
            (0, 3, 1.42j),
            (2, 5, 1.42j),
            (4, 7, 1.42j),
            (6, 9, 1.42j),
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
            (3, 3, 1.0),
            (4, 4, 1.0),
            (5, 5, 1.0),
            (6, 6, 1.0),
            (7, 7, 1.0),
            (8, 8, 1.0),
            (9, 9, 1.0),
        ]

        hexa = HexagonalLattice(rows, cols, edge_parameter, onsite_parameter)

        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(10))
            target_graph.add_edges_from(weighted_edge_list)
            self.assertTrue(
                is_isomorphic(hexa.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(hexa.num_nodes, 10)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(hexa.node_indexes), set(range(10)))

        with self.subTest("Check the set of weights."):
            target_set = set(weighted_edge_list)
            self.assertSetEqual(set(hexa.weighted_edge_list), target_set)

        with self.subTest("Check the adjacency matrix."):
            target_matrix = np.zeros((10, 10), dtype=complex)

            indices = [(a, b) for a, b, _ in weighted_edge_list]

            for idx1, idx2 in indices:
                target_matrix[idx1, idx2] = 0 + 1.42j

            target_matrix -= target_matrix.T

            np.fill_diagonal(target_matrix, 1.0)

            assert_array_equal(hexa.to_adjacency_matrix(weighted=True), target_matrix)
